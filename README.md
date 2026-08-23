# flmgm-sft FeatureCloud App

![downloads](./badges/downloads.svg?v=423 "Cumulative GitHub full clones tracked by this repository workflow")

This app adapts the FL-MGM SFT classification workflow to the FeatureCloud app model.

## What It Does

- Runs local training on each client.
- Aggregates model updates on the coordinator.
- Iterates for multiple rounds.

## Get MGM as based model to test
```
git clone https://github.com/wen1112/flmgm.git
git clone https://github.com/HUST-NingKang-Lab/MGM.git
cp -r MGM/mgm ./flmgm

```

## Key Files

- states.py: FeatureCloud state machine and training logic
- config.yml: runtime configuration (place in /mnt/input)

## Config

Create a config file at /mnt/input/config.yml.

Example:

```yaml
input_base: /mnt/input
label_column: label
label_values: ["0", "1"]
num_rounds: 3
local_epochs: 1
batch_size: 8
learning_rate: 5e-5
weight_decay: 0.01
warmup_ratio: 0.1
lr_scheduler_type: linear
train_val_split: 0.9
client_metadata_file: metadata.csv
data_file: data.csv
global_val_metadata_file: global_val_metadata.csv
model_path: null
outputs_base_dir: /mnt/output
use_dp: false
use_smpc: false
seed: 42
```

## Data Layout

Each client container sees:

- /mnt/input/data.csv
- /mnt/input/metadata.csv
- /mnt/input/config.yml

If using a shared validation set, place it in the generic dir so it appears in /mnt/input on each client.

## Build and Test

```bash
featurecloud app build ./flmgm-sft flmgm-sft
featurecloud controller start --port=8000 --data-dir=./data
featurecloud test start \
  --controller-host=http://localhost:8000 \
  --client-dirs=./data/client1,./data/client2 \
  --generic-dir=./data/generic_dir \
  --app-image=flmgm-sft \
  --query-interval=1
```
