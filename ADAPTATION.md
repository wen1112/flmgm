# FL-MGM SFT FeatureCloud Adaptation Notes

This document lists the concrete changes made to adapt the mock FL script into a FeatureCloud app.

## 1) State Machine Mapping

Original script flow:
- load data
- local train per client
- aggregate
- repeat for multiple rounds

FeatureCloud states:
- initial: load config, tokenizer, label map, and initialize global model
- local_train: train on client data and send state to coordinator
- aggregate: coordinator aggregates and broadcasts new global model
- wait_global: participants wait for updated model
- terminal: stop after final round

## 2) Data and Config Adaptation

FeatureCloud input/output constraints:
- Inputs must be read from /mnt/input
- Outputs should be written to /mnt/output

New config file:
- /mnt/input/config.yml is required
- config.example.yml shows all supported keys

Key config keys:
- label_column
- data_file / client_metadata_file
- num_rounds / local_epochs / batch_size
- model_path (optional)

## 3) Model and Tokenizer

- The tokenizer and model are loaded from MGM resources.
- A fallback import path is added if mgm is vendored into the app directory.

If mgm is not installed:
- Copy the mgm package into this app folder (so /app/mgm exists in the container).

## 4) Communication

- Clients send state_dict payloads via send_data_to_coordinator.
- Coordinator uses gather_data to collect updates.
- Aggregation is weighted by local sample counts.
- Coordinator broadcasts the updated global model each round.

## 5) Privacy Hooks

- FeatureCloud DP/SMPC hooks are wired via use_dp/use_smpc in config.
- These are optional and default to false.

Note: DP/SMPC over large state_dict payloads can be expensive. Use with care.

## 6) Dependencies

requirements.txt now includes:
- torch, transformers, pandas, numpy, scikit-learn, matplotlib, bios

## 7) Files Changed

- states.py: replaced template demo with FL training states
- requirements.txt: added ML dependencies
- README.md: updated app description and usage
- config.example.yml: config template
- ADAPTATION.md: this document
