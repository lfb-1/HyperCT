# HyperCT: Low-Rank Hypernet for Unified Chest CT Analysis

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```text
ct2echo_pkg/
├── conf/                    # Hydra configuration
│   ├── config.yaml          # Main config
│   ├── model/dinov3.yaml    # Model config
│   ├── training/            # Training hyperparameters
│   └── azureml/             # Azure ML settings
├── ct2echo/                 # Core package
│   ├── models/              # DINOv3 + Hypernet
│   ├── training/            # Training logic
│   ├── evaluation/          # Evaluation metrics
│   └── data/                # Data loading
└── scripts/                 # Entry points
```

## Quick Start

### Local Training

```bash
cd scripts

# Run with default config
python mtl_main_hydra.py

```

### Azure ML Training

```bash
cd scripts

# Dry run (validate config)
python submit_job_hydra.py dry_run=true

# Submit training job
python submit_job_hydra.py

# Debug mode (enables VS Code + JupyterLab)
python submit_job_hydra.py debug_mode=true

```

## Configuration

Before running on Azure ML, update `conf/azureml/production.yaml`:

```yaml
subscription_id: "<YOUR_SUBSCRIPTION_ID>"
resource_group: "<YOUR_RESOURCE_GROUP>"
workspace_name: "<YOUR_WORKSPACE_NAME>"
compute_name: "<YOUR_COMPUTE_NAME>"
environment_name: "<YOUR_ENVIRONMENT_NAME>"
experiment_name: "<YOUR_EXPERIMENT_NAME>"
data_base_path: "Users/<YOUR_USERNAME>/data_weights"
```

## Key Parameters

| Parameter                        | Default | Description              |
| -------------------------------- | ------- | ------------------------ |
| `model.lora_rank`                | 16      | LoRA adapter rank        |
| `model.learnable_embedding_dim`  | 512     | Task embedding dimension |
| `training.lr`                    | 1e-5    | Learning rate            |
| `training.num_epochs`            | 20      | Number of epochs         |
| `training.batch_size`            | 8       | Batch size               |

## Output

Training outputs are saved to `paths.output_dir`:

- `epoch_N/` - Checkpoints per epoch
- `final/` - Final model weights
- `hydra_config.yaml` - Saved configuration

## Saliency Maps

Pre-computed saliency maps are available on [Google Drive](https://drive.google.com/drive/folders/1lEtJkG8wuZbQGd56rAQDYt7MvvZ7Xnap?usp=drive_link).
