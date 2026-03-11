"""
FeatureCloud app states for FL-MGM SFT classification.

State flow:
- initial -> local_train
- local_train -> aggregate (coordinator) or wait_global (participants)
- wait_global -> local_train
- aggregate -> local_train or terminal
"""

import os
import sys
import copy
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset
from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    roc_auc_score = None
    SKLEARN_AVAILABLE = False

from FeatureCloud.app.engine.app import AppState, app_state, Role

try:
    from mgm.CLI.CLI_utils import get_CFG_reader, find_pkg_resource
    from mgm.src.MicroCorpus import MicroCorpus
    from mgm.src.utils import seed_everything, CustomUnpickler
except ImportError:
    # Allow local development when mgm is vendored into the app folder
    mgm_path = os.path.join(APP_ROOT, "mgm")
    if os.path.isdir(mgm_path):
        sys.path.insert(0, APP_ROOT)
        from mgm.CLI.CLI_utils import get_CFG_reader, find_pkg_resource
        from mgm.src.MicroCorpus import MicroCorpus
        from mgm.src.utils import seed_everything, CustomUnpickler
    else:
        raise

try:
    import bios
except ImportError:
    bios = None


DEFAULT_CONFIG_PATHS = [
    "/mnt/input/config.yml",
    "/app/config.yml",
]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, record: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_runtime_dependencies(outputs_base_dir: str, logger=None):
    """Export pip freeze from the runtime container."""
    deps_dir = os.path.join(outputs_base_dir, "env")
    ensure_dir(deps_dir)
    out_path = os.path.join(deps_dir, "pip_freeze.txt")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        if logger:
            logger(f"[{now_ts()}] [initial] Runtime deps exported to {out_path}")
    except Exception as exc:
        if logger:
            logger(f"[{now_ts()}] [initial] Runtime deps export failed: {exc}")
    return out_path


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def read_config() -> Dict:
    if bios is None:
        raise ImportError("bios is required to read config.yml")

    for path in DEFAULT_CONFIG_PATHS:
        if os.path.exists(path):
            return bios.read(path)

    raise FileNotFoundError("config.yml not found in /mnt/input or /app")


def load_tokenizer():
    with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
        unpickler = CustomUnpickler(f)
        tokenizer = unpickler.load()
    return tokenizer


def build_label_map(label_column: str, label_values: List[str], metadata_paths: List[str]) -> Tuple[Dict[str, int], int]:
    # label_values 用于固定标签空间顺序，避免不同客户端/轮次的标签编码不一致
    if label_values:
        sorted_values = [str(v) for v in label_values]
    else:
        unique_values = set()
        for metadata_file in metadata_paths:
            if not os.path.exists(metadata_file):
                continue
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            if label_column not in metadata_df.columns:
                continue
            values = metadata_df[label_column].dropna().astype(str).tolist()
            unique_values.update(values)
        sorted_values = sorted(unique_values)

    label_map = {value: idx for idx, value in enumerate(sorted_values)}
    return label_map, len(label_map)


def resolve_model_path(raw_path: str) -> str:
    """Resolve model path from config; supports app-relative paths."""
    if not raw_path:
        return find_pkg_resource("resources/general_model")

    # Treat relative paths as app-root relative
    model_path = raw_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(APP_ROOT, model_path)

    # If a model directory is provided directly, use it
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    # Otherwise, assume a package root and append resources/general_model
    candidate = os.path.join(model_path, "resources", "general_model")
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return candidate

    # Fall back to provided path
    return model_path


def encode_singlelabel_targets(sample_ids: List[str], metadata_df: pd.DataFrame, label_column: str, label_map: Dict[str, int]) -> np.ndarray:
    labels = np.full((len(sample_ids),), fill_value=-100, dtype=np.int64)

    metadata_df = metadata_df.copy()
    metadata_df.index = metadata_df.index.astype(str)
    aligned_metadata = metadata_df.reindex(sample_ids)

    for row_idx, sample_id in enumerate(sample_ids):
        row = aligned_metadata.iloc[row_idx]
        value = row[label_column]
        if pd.isna(value):
            continue
        value_str = str(value)
        if value_str not in label_map:
            continue
        labels[row_idx] = label_map[value_str]

    return labels


def stratified_split_indices(labels: np.ndarray, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    valid_indices = np.where(labels >= 0)[0].tolist()
    total_size = len(valid_indices)
    if total_size < 2:
        return valid_indices, []

    grouped_indices: Dict[int, List[int]] = {}
    for idx in valid_indices:
        grouped_indices.setdefault(int(labels[idx]), []).append(idx)

    target_train_size = int(round(total_size * train_ratio))
    target_train_size = max(1, min(target_train_size, total_size - 1))

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, idxs in grouped_indices.items():
        local = idxs[:]
        rng.shuffle(local)
        n = len(local)

        if n == 1:
            train_indices.extend(local)
            continue

        n_train = int(round(n * train_ratio))
        n_train = max(1, min(n_train, n - 1))
        train_indices.extend(local[:n_train])
        val_indices.extend(local[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    while len(train_indices) > target_train_size and train_indices:
        val_indices.append(train_indices.pop())
    while len(train_indices) < target_train_size and val_indices:
        train_indices.append(val_indices.pop())

    if len(val_indices) == 0 and len(train_indices) > 1:
        val_indices.append(train_indices.pop())

    return train_indices, val_indices


def compute_multiclass_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=1).astype(np.int64)
    trues = labels.astype(np.int64)
    valid_mask = trues >= 0
    preds = preds[valid_mask]
    trues = trues[valid_mask]

    if trues.size == 0:
        return {
            "accuracy": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "roc_auc_ovr_macro": float("nan"),
            "roc_auc_ovr_weighted": float("nan"),
        }

    num_classes = int(max(np.max(preds), np.max(trues))) + 1
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(trues, preds):
        conf[t, p] += 1

    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0).astype(np.float64) - tp
    fn = conf.sum(axis=1).astype(np.float64) - tp
    eps = 1e-12

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    accuracy = float((preds == trues).mean())
    macro_precision = float(np.mean(precision_per_class))
    macro_recall = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))

    roc_auc_ovr_macro = float("nan")
    roc_auc_ovr_weighted = float("nan")
    if SKLEARN_AVAILABLE and num_classes > 1 and trues.size > 1:
        try:
            max_logits = np.max(logits[valid_mask], axis=1, keepdims=True)
            exp_logits = np.exp(logits[valid_mask] - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            y_true_onehot = np.eye(num_classes, dtype=np.int32)[trues]
            roc_auc_ovr_macro = float(
                roc_auc_score(y_true_onehot, probs, multi_class="ovr", average="macro")
            )
            roc_auc_ovr_weighted = float(
                roc_auc_score(y_true_onehot, probs, multi_class="ovr", average="weighted")
            )
        except ValueError:
            pass

    return {
        "accuracy": accuracy,
        "micro_f1": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
    }


def build_compute_metrics_fn():
    def _compute_metrics(eval_pred):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        return compute_multiclass_metrics(logits, labels)
    return _compute_metrics


def load_client_data(input_base: str, data_file: str, metadata_file: str, tokenizer, cfg, label_column: str, label_map: Dict[str, int], train_val_split: float, seed: int):
    # 读取客户端数据与标签，并做分层划分
    data_path = os.path.join(input_base, data_file)
    metadata_path = os.path.join(input_base, metadata_file)

    corpus = MicroCorpus(
        data_path=data_path,
        tokenizer=tokenizer,
        max_len=cfg.getint("construct", "max_len"),
        preprocess=True,
    )

    metadata_df = pd.read_csv(metadata_path, index_col=0)
    sample_ids = corpus.data.index.astype(str).tolist()
    labels = encode_singlelabel_targets(sample_ids, metadata_df, label_column, label_map)

    dataset = TextDataset(
        input_ids=corpus[:]["input_ids"],
        attention_mask=corpus[:]["attention_mask"],
        labels=labels,
    )

    train_indices, val_indices = stratified_split_indices(labels, train_val_split, seed)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


def load_global_val(input_base: str, data_file: str, metadata_file: str, tokenizer, cfg, label_column: str, label_map: Dict[str, int]):
    # 读取全局验证集
    data_path = os.path.join(input_base, data_file)
    metadata_path = os.path.join(input_base, metadata_file)

    corpus = MicroCorpus(
        data_path=data_path,
        tokenizer=tokenizer,
        max_len=cfg.getint("construct", "max_len"),
        preprocess=True,
    )

    metadata_df = pd.read_csv(metadata_path, index_col=0)
    sample_ids = corpus.data.index.astype(str).tolist()
    labels = encode_singlelabel_targets(sample_ids, metadata_df, label_column, label_map)

    return TextDataset(
        input_ids=corpus[:]["input_ids"],
        attention_mask=corpus[:]["attention_mask"],
        labels=labels,
    )


def create_model(model_path: str, num_labels: int, tokenizer):
    model = GPT2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def to_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def average_state_dicts(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    total_weight = float(sum(weights))
    avg_state = {}
    for key in states[0]:
        weighted_sum = sum(state[key] * weight for state, weight in zip(states, weights))
        avg_state[key] = weighted_sum / total_weight
    return avg_state


def evaluate_model(model, eval_set, outputs_base_dir: str):
    # 统一的评估入口，便于输出一致指标
    eval_log_dir = os.path.join(outputs_base_dir, "eval_logs")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=eval_log_dir,
            per_device_eval_batch_size=1,
            no_cuda=False if torch.cuda.is_available() else True,
            report_to="none",
        ),
        eval_dataset=eval_set,
        compute_metrics=build_compute_metrics_fn(),
    )
    return trainer.evaluate()


class FedProxTrainer(Trainer):
    """FedProx: add proximal term to local loss."""

    def __init__(self, *args, mu=0.0, global_model_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.global_model_params = [p.clone().detach() for p in global_model_params] if global_model_params else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits, labels.long())

        if self.mu > 0 and self.global_model_params is not None:
            proximal_term = 0.0
            for param_curr, param_glob in zip(model.parameters(), self.global_model_params):
                param_glob = param_glob.to(param_curr.device)
                proximal_term += torch.sum((param_curr - param_glob) ** 2)
            loss += (self.mu / 2) * proximal_term

        return (loss, outputs) if return_outputs else loss


def save_metrics(outputs_base_dir: str, filename: str, record: Dict):
    metrics_dir = os.path.join(outputs_base_dir, "metrics")
    ensure_dir(metrics_dir)
    path = os.path.join(metrics_dir, filename)
    append_jsonl(path, record)
    return path


def save_timing(outputs_base_dir: str, record: Dict):
    return save_metrics(outputs_base_dir, "timing.jsonl", record)


def plot_metrics(outputs_base_dir: str, filename: str, rounds: List[int], series: Dict[str, List[float]], title: str, ylabel: str):
    plots_dir = os.path.join(outputs_base_dir, "plots")
    ensure_dir(plots_dir)
    path = os.path.join(plots_dir, filename)

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(rounds, values, marker="o", linewidth=2, markersize=5, label=label)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def safe_send_to_coordinator(app_state: AppState, payload: Dict, use_dp: bool, use_smpc: bool, memo: str):
    try:
        app_state.send_data_to_coordinator(
            payload, send_to_self=True, use_dp=use_dp, use_smpc=use_smpc, memo=memo
        )
    except TypeError:
        app_state.send_data_to_coordinator(payload, send_to_self=True, use_dp=use_dp, memo=memo)


def safe_await_data(app_state: AppState, use_dp: bool, use_smpc: bool, memo: str):
    try:
        return app_state.await_data(use_dp=use_dp, use_smpc=use_smpc, memo=memo)
    except TypeError:
        return app_state.await_data(use_dp=use_dp, memo=memo)


def safe_gather_data(app_state: AppState, use_dp: bool, use_smpc: bool, memo: str):
    try:
        return app_state.gather_data(use_dp=use_dp, use_smpc=use_smpc, memo=memo)
    except TypeError:
        return app_state.gather_data(use_dp=use_dp, memo=memo)


def safe_broadcast_data(app_state: AppState, payload: Dict, use_dp: bool, use_smpc: bool, memo: str):
    try:
        app_state.broadcast_data(
            payload, send_to_self=True, use_dp=use_dp, use_smpc=use_smpc, memo=memo
        )
    except TypeError:
        app_state.broadcast_data(payload, send_to_self=True, use_dp=use_dp, memo=memo)


@app_state("initial")
class InitialState(AppState):
    def register(self):
        self.register_transition("local_train", Role.BOTH)

    def run(self):
        cfg = read_config()

        self.log(f"[{now_ts()}] [initial] Config loaded")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.log(
                f"[{now_ts()}] [initial] CUDA available: True, device_count={torch.cuda.device_count()}, "
                f"device_name={torch.cuda.get_device_name(0)}"
            )
        else:
            self.log(f"[{now_ts()}] [initial] CUDA available: False")

        export_runtime_dependencies(cfg.get("outputs_base_dir", "/mnt/output"), logger=self.log)

        seed = int(cfg.get("seed", 42))
        seed_everything(seed)

        if cfg.get("use_dp"):
            self.configure_dp(
                epsilon=float(cfg.get("dp_epsilon", 1.0)),
                delta=float(cfg.get("dp_delta", 1e-5)),
                clippingVal=float(cfg.get("dp_clip_norm", 1.0)),
            )
            self.log(f"[{now_ts()}] [initial] DP enabled")
        if cfg.get("use_smpc"):
            self.configure_smpc(
                exponent=int(cfg.get("smpc_exponent", 8)),
                shards=int(cfg.get("smpc_shards", 0)),
            )
            self.log(f"[{now_ts()}] [initial] SMPC enabled")

        tokenizer = load_tokenizer()
        cfg_reader = get_CFG_reader()

        label_column = cfg.get("label_column")
        if not label_column:
            raise ValueError("label_column is required in config.yml")

        input_base = cfg.get("input_base", "/mnt/input")
        data_file = cfg.get("data_file", "data.csv")
        metadata_file = cfg.get("client_metadata_file", "metadata.csv")
        global_val_metadata_file = cfg.get("global_val_metadata_file")
        global_val_file = cfg.get("global_val_file")

        self.log(f"[{now_ts()}] [initial] input_base={input_base}")
        self.log(f"[{now_ts()}] [initial] data_file={data_file}")
        self.log(f"[{now_ts()}] [initial] client_metadata_file={metadata_file}")
        if global_val_file and global_val_metadata_file:
            self.log(f"[{now_ts()}] [initial] global_val_file={global_val_file}")
            self.log(f"[{now_ts()}] [initial] global_val_metadata_file={global_val_metadata_file}")

        metadata_paths = [os.path.join(input_base, metadata_file)]
        if global_val_metadata_file:
            metadata_paths.append(os.path.join(input_base, global_val_metadata_file))

        label_values = cfg.get("label_values", [])
        label_map, num_labels = build_label_map(label_column, label_values, metadata_paths)

        model_path = resolve_model_path(cfg.get("model_path"))
        self.log(f"[{now_ts()}] [initial] model_path={model_path}")

        model = create_model(model_path, num_labels, tokenizer)
        global_state = to_cpu_state_dict(model.state_dict())

        self.store("config", cfg)
        self.store("cfg_reader", cfg_reader)
        self.store("tokenizer", tokenizer)
        self.store("label_map", label_map)
        self.store("num_labels", num_labels)
        self.store("global_state", global_state)
        self.store("round", 0)
        # 全局早停初始化
        self.store("best_global_loss", float("inf"))
        self.store("no_improve_rounds", 0)
        if global_val_file and global_val_metadata_file:
            self.store("global_val_file", global_val_file)
            self.store("global_val_metadata_file", global_val_metadata_file)
        self.log(f"[{now_ts()}] [initial] Initialization complete")

        return "local_train"


@app_state("local_train")
class LocalTrainState(AppState):
    def register(self):
        self.register_transition("aggregate", Role.COORDINATOR)
        self.register_transition("wait_global", Role.PARTICIPANT)

    def run(self):
        cfg = self.load("config")
        cfg_reader = self.load("cfg_reader")
        tokenizer = self.load("tokenizer")
        label_map = self.load("label_map")
        num_labels = self.load("num_labels")
        global_state = self.load("global_state")
        round_idx = self.load("round")

        input_base = cfg.get("input_base", "/mnt/input")
        data_file = cfg.get("data_file", "data.csv")
        metadata_file = cfg.get("client_metadata_file", "metadata.csv")

        t0 = time.perf_counter()
        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: loading client data")
        train_set, val_set = load_client_data(
            input_base=input_base,
            data_file=data_file,
            metadata_file=metadata_file,
            tokenizer=tokenizer,
            cfg=cfg_reader,
            label_column=cfg["label_column"],
            label_map=label_map,
            train_val_split=float(cfg.get("train_val_split", 0.9)),
            seed=int(cfg.get("seed", 42)),
        )

        model_path = resolve_model_path(cfg.get("model_path"))

        data_time = time.perf_counter() - t0
        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: train={len(train_set)} val={len(val_set)} data_time={data_time:.2f}s")
        model = create_model(model_path, num_labels, tokenizer)
        model.load_state_dict(global_state)

        training_args = TrainingArguments(
            output_dir=os.path.join(cfg.get("outputs_base_dir", "/mnt/output"), "client_checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=int(cfg.get("local_epochs", 1)),
            per_device_train_batch_size=int(cfg.get("batch_size", 8)),
            learning_rate=float(cfg.get("learning_rate", 5e-5)),
            weight_decay=float(cfg.get("weight_decay", 0.01)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            lr_scheduler_type=str(cfg.get("lr_scheduler_type", "linear")),
            logging_strategy="steps",
            logging_steps=5,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            no_cuda=False if torch.cuda.is_available() else True,
        )

        mu = float(cfg.get("mu", 0.0))
        global_model_for_prox = create_model(model_path, num_labels, tokenizer)
        global_model_for_prox.load_state_dict(global_state)

        trainer = FedProxTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=build_compute_metrics_fn(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("local_early_stopping_patience", 1)))],
            mu=mu,
            global_model_params=list(global_model_for_prox.parameters()),
        )

        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: training start")
        t1 = time.perf_counter()
        trainer.train()
        train_time = time.perf_counter() - t1
        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: training complete in {train_time:.2f}s")
        local_state = to_cpu_state_dict(trainer.model.state_dict())

        t2 = time.perf_counter()
        eval_result = trainer.evaluate(eval_dataset=val_set)
        eval_time = time.perf_counter() - t2
        if isinstance(eval_result, dict):
            loss = eval_result.get("eval_loss")
            acc = eval_result.get("eval_accuracy")
            f1 = eval_result.get("eval_macro_f1")
            self.log(f"[{now_ts()}] [local_train] Round {round_idx}: eval_loss={loss} eval_acc={acc} eval_macro_f1={f1} eval_time={eval_time:.2f}s")

            outputs_base = cfg.get("outputs_base_dir", "/mnt/output")
            record = {
                "ts": now_ts(),
                "round": int(round_idx),
                "role": "coordinator" if self.is_coordinator else "participant",
                "client_id": str(self.id),
                "eval_loss": loss,
                "eval_accuracy": acc,
                "eval_macro_f1": f1,
            }
            save_metrics(outputs_base, "local_eval.jsonl", record)

            timing_record = {
                "ts": now_ts(),
                "round": int(round_idx),
                "stage": "local_train",
                "role": "coordinator" if self.is_coordinator else "participant",
                "client_id": str(self.id),
                "data_time_sec": data_time,
                "train_time_sec": train_time,
                "eval_time_sec": eval_time,
            }
            save_timing(outputs_base, timing_record)

        payload = {
            "round": round_idx,
            "num_samples": len(train_set),
            "state": local_state,
        }

        use_dp = bool(cfg.get("use_dp", False))
        use_smpc = bool(cfg.get("use_smpc", False))
        t3 = time.perf_counter()
        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: sending update to coordinator")
        safe_send_to_coordinator(self, payload, use_dp=use_dp, use_smpc=use_smpc, memo=f"round-{round_idx}")
        send_time = time.perf_counter() - t3
        self.log(f"[{now_ts()}] [local_train] Round {round_idx}: update sent in {send_time:.2f}s")

        if self.is_coordinator:
            return "aggregate"
        return "wait_global"


@app_state("wait_global")
class WaitGlobalState(AppState):
    def register(self):
        self.register_transition("local_train", Role.PARTICIPANT)
        self.register_transition("terminal", Role.PARTICIPANT)

    def run(self):
        cfg = self.load("config")
        round_idx = self.load("round")

        t0 = time.perf_counter()
        self.log(f"[{now_ts()}] [wait_global] Round {round_idx}: waiting for global model")
        payload = safe_await_data(
            self,
            use_dp=bool(cfg.get("use_dp", False)),
            use_smpc=bool(cfg.get("use_smpc", False)),
            memo=f"round-{round_idx + 1}",
        )
        wait_time = time.perf_counter() - t0
        if payload.get("stop"):
            self.log(f"[{now_ts()}] [wait_global] Stop signal received")
            return "terminal"

        self.store("global_state", payload["state"])
        self.store("round", int(payload["round"]))
        outputs_base = cfg.get("outputs_base_dir", "/mnt/output")
        save_timing(outputs_base, {
            "ts": now_ts(),
            "round": int(payload.get("round", round_idx + 1)),
            "stage": "wait_global",
            "role": "participant",
            "client_id": str(self.id),
            "wait_time_sec": wait_time,
        })
        return "local_train"


@app_state("aggregate")
class AggregateState(AppState):
    def register(self):
        self.register_transition("local_train", Role.COORDINATOR)
        self.register_transition("terminal", Role.COORDINATOR)

    def run(self):
        cfg = self.load("config")
        round_idx = self.load("round")

        use_dp = bool(cfg.get("use_dp", False))
        use_smpc = bool(cfg.get("use_smpc", False))

        t0 = time.perf_counter()
        self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: gathering client updates")
        data_list = safe_gather_data(self, use_dp=use_dp, use_smpc=use_smpc, memo=f"round-{round_idx}")
        gather_time = time.perf_counter() - t0

        states = [item["state"] for item in data_list]
        agg_strategy = str(cfg.get("aggregation_strategy", "weighted")).lower()
        if agg_strategy == "uniform":
            weights = [1.0 for _ in data_list]
        else:
            weights = [float(item.get("num_samples", 1.0)) for item in data_list]

        t1 = time.perf_counter()
        self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: aggregating {len(states)} updates")
        global_state = average_state_dicts(states, weights)
        agg_time = time.perf_counter() - t1
        self.store("global_state", global_state)

        next_round = int(round_idx) + 1
        total_rounds = int(cfg.get("num_rounds", 1))

        if self.is_coordinator and cfg.get("global_val_file") and cfg.get("global_val_metadata_file"):
            try:
                input_base = cfg.get("input_base", "/mnt/input")
                global_val_set = load_global_val(
                    input_base=input_base,
                    data_file=cfg.get("global_val_file"),
                    metadata_file=cfg.get("global_val_metadata_file"),
                    tokenizer=self.load("tokenizer"),
                    cfg=self.load("cfg_reader"),
                    label_column=cfg.get("label_column"),
                    label_map=self.load("label_map"),
                )
                model_path = resolve_model_path(cfg.get("model_path"))
                eval_model = create_model(model_path, self.load("num_labels"), self.load("tokenizer"))
                eval_model.load_state_dict(global_state)
                t2 = time.perf_counter()
                eval_result = evaluate_model(eval_model, global_val_set, cfg.get("outputs_base_dir", "/mnt/output"))
                eval_time = time.perf_counter() - t2
                self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: global_val={eval_result} eval_time={eval_time:.2f}s")

                outputs_base = cfg.get("outputs_base_dir", "/mnt/output")
                record = {
                    "ts": now_ts(),
                    "round": int(round_idx),
                    "eval_loss": eval_result.get("eval_loss"),
                    "eval_accuracy": eval_result.get("eval_accuracy"),
                    "eval_macro_f1": eval_result.get("eval_macro_f1"),
                }
                save_metrics(outputs_base, "global_val.jsonl", record)
                save_timing(outputs_base, {
                    "ts": now_ts(),
                    "round": int(round_idx),
                    "stage": "aggregate",
                    "role": "coordinator",
                    "gather_time_sec": gather_time,
                    "aggregate_time_sec": agg_time,
                    "global_val_time_sec": eval_time,
                })

                # 全局早停判定（以 global_val 的 eval_loss 为准）
                best_loss = float(self.load("best_global_loss"))
                no_improve = int(self.load("no_improve_rounds"))
                patience = int(cfg.get("global_early_stopping_patience", 0))
                min_delta = float(cfg.get("global_min_delta", 0.0))

                curr_loss = eval_result.get("eval_loss")
                if curr_loss is not None:
                    if curr_loss < (best_loss - min_delta):
                        best_loss = float(curr_loss)
                        no_improve = 0
                        self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: global_val improved to {best_loss:.6f}")
                    else:
                        no_improve += 1
                        self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: no global_val improvement ({no_improve}/{patience})")

                    self.store("best_global_loss", best_loss)
                    self.store("no_improve_rounds", no_improve)

                rounds = []
                losses = []
                accs = []
                f1s = []
                metrics_path = os.path.join(outputs_base, "metrics", "global_val.jsonl")
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        for line in f:
                            item = json.loads(line)
                            rounds.append(int(item.get("round", 0)))
                            losses.append(item.get("eval_loss"))
                            accs.append(item.get("eval_accuracy"))
                            f1s.append(item.get("eval_macro_f1"))
                plot_metrics(outputs_base, "global_val_loss.png", rounds, {"loss": losses}, "Global Val Loss", "Loss")
                plot_metrics(outputs_base, "global_val_acc.png", rounds, {"accuracy": accs}, "Global Val Accuracy", "Accuracy")
                plot_metrics(outputs_base, "global_val_macro_f1.png", rounds, {"macro_f1": f1s}, "Global Val Macro F1", "Macro F1")
            except Exception as exc:
                self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: global_val failed: {exc}")

        # 早停条件：达到轮数或触发全局早停
        patience = int(cfg.get("global_early_stopping_patience", 0))
        no_improve = int(self.load("no_improve_rounds"))
        early_stop = patience > 0 and no_improve >= patience

        if next_round >= total_rounds or early_stop:
            payload = {"round": next_round, "state": global_state, "stop": True}
            safe_broadcast_data(self, payload, use_dp=use_dp, use_smpc=use_smpc, memo=f"round-{next_round}")
            if early_stop:
                self.log(f"[{now_ts()}] [aggregate] Global early stop triggered (patience={patience})")
            self.log(f"[{now_ts()}] [aggregate] Training complete, broadcasting stop")
            return "terminal"

        t3 = time.perf_counter()
        payload = {"round": next_round, "state": global_state, "stop": False}
        safe_broadcast_data(self, payload, use_dp=use_dp, use_smpc=use_smpc, memo=f"round-{next_round}")
        broadcast_time = time.perf_counter() - t3
        self.store("round", next_round)
        self.log(f"[{now_ts()}] [aggregate] Round {round_idx}: broadcasted next round {next_round} in {broadcast_time:.2f}s")
        return "local_train"
