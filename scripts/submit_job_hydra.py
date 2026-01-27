#!/usr/bin/env python3
"""
Azure ML Job Submission Script for CT2ECHO using Hydra Configuration

Usage:
  # Production training with DINOv3 hypernet model
  python submit_job_hydra.py

  # Debug mode (adds debugpy, VS Code, and JupyterLab services)
  python submit_job_hydra.py debug_mode=true

  # Override specific training parameters
  python submit_job_hydra.py training.lr=0.001 training.num_epochs=5

  # Dry run to validate configuration
  python submit_job_hydra.py dry_run=true
"""

import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from azure.ai.ml import MLClient, command, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

try:
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("Hydra not found. Please install with: pip install hydra-core")
    sys.exit(1)

import csv

# Add package root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ct2echo.utils.azure_logging import quiet_azure_logs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================== HELPER FUNCTIONS ========================


def create_ml_client(azureml_cfg) -> MLClient:
    """Create Azure ML client"""
    try:
        with quiet_azure_logs():
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=azureml_cfg.subscription_id,
                resource_group_name=azureml_cfg.resource_group,
                workspace_name=azureml_cfg.workspace_name,
            )

        logger.info(f"Connected to Azure ML workspace: {ml_client.workspace_name}")
        return ml_client
    except Exception as e:
        logger.error(f"Failed to create ML client: {e}")
        raise


def build_base_command(debug_mode: bool = False) -> list:
    """Build base command with optional debugging"""
    if debug_mode:
        return ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m"]
    return ["python"]


def create_services(debug_mode: bool = False) -> dict:
    """Create job services"""
    services = {}
    if debug_mode:
        from azure.ai.ml.entities import VsCodeJobService, JupyterLabJobService

        services["vscode"] = VsCodeJobService(nodes="all")
        services["jupyterlab"] = JupyterLabJobService(nodes="all")

    return services


def create_training_outputs(cfg: DictConfig, model_type: str) -> dict:
    """Create output definitions for training jobs"""
    azureml_cfg = cfg.azureml
    base_path = f"{azureml_cfg.path_prefix}/{azureml_cfg.data_base_path}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = azureml_cfg.output_path_template.format(model_type=model_type, timestamp=timestamp)

    return {
        "save_weight_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{output_folder}"),
        "task_weight_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{azureml_cfg.task_embed_path}"),
        "train_val_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{azureml_cfg.train_val_path}"),
        "pretrain_weight_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{azureml_cfg.pretrain_weight_path}"),
        "dinov3_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{azureml_cfg.dinov3_path}"),
        "checkpoint_dir": Output(type=AssetTypes.URI_FOLDER, path=f"{base_path}/{azureml_cfg.checkpoint_path}"),
    }


def create_hydra_overrides_command(cfg: DictConfig) -> str:
    """Create command with Hydra overrides for Azure ML paths and model config"""
    overrides = []

    model_choice = None
    try:
        runtime_choices = HydraConfig.get().runtime.choices
        model_choice = runtime_choices.get("model") if runtime_choices else None
    except Exception:
        model_choice = None

    # Add model selection override (DINOv3 only)
    if model_choice:
        overrides.append(f"model={model_choice}")
    else:
        overrides.append("model=dinov3")

    # Add path overrides
    overrides.extend(
        [
            "paths.output_dir=${outputs.save_weight_dir}",
            "paths.pretrain_weight_dir=${outputs.pretrain_weight_dir}",
            "paths.train_val_dir=${outputs.train_val_dir}",
            "paths.task_embed_dir=${outputs.task_weight_dir}",
            "paths.dinov3_dir=${outputs.dinov3_dir}",
        ]
    )
    return " ".join(overrides)


# ======================== JOB CREATORS ========================


def create_training_job(cfg: DictConfig, debug_mode: bool = False, display_name: Optional[str] = None):
    """Create training job using Hydra configuration"""

    azureml_cfg = cfg.azureml

    # DINOv3 hypernet model type
    model_type = "dinov3_hypernet"
    job_prefix = "dinov3"

    if not display_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "debug" if debug_mode else "train"
        display_name = f"ct2echo_{job_prefix}_{mode}_{timestamp}"

    # Build Hydra command with path overrides
    command_parts = build_base_command(debug_mode)
    hydra_overrides = create_hydra_overrides_command(cfg)
    command_parts.extend(["scripts/mtl_main_hydra.py", hydra_overrides])

    # Use debug experiment name if in debug mode (append _DEBUG to any experiment)
    experiment_name = azureml_cfg.experiment_name
    if debug_mode and not experiment_name.endswith("_DEBUG"):
        experiment_name = f"{experiment_name}_DEBUG"

    return command(
        code=".",
        command=" ".join(command_parts),
        experiment_name=experiment_name,
        environment=azureml_cfg.environment_name,
        compute=azureml_cfg.compute_name,
        display_name=display_name,
        environment_variables=dict(azureml_cfg.job_env_vars) if azureml_cfg.job_env_vars else {},
        services=create_services(debug_mode),
        outputs=create_training_outputs(cfg, model_type),
    )


# ======================== MAIN SUBMISSION FUNCTIONS ========================


def submit_job(
    cfg: DictConfig, job_type: str = "training", debug_mode: bool = False, display_name: Optional[str] = None
):
    """Submit job based on Hydra configuration"""

    # Create appropriate job
    if job_type == "training":
        job = create_training_job(cfg, debug_mode, display_name)
        job_description = f"Training ({get_model_description(cfg)})"
    else:
        raise ValueError(f"Unknown job type: {job_type}")

    # Submit job
    ml_client = create_ml_client(cfg.azureml)

    # Suppress Azure SDK logs during job creation and submission
    original_azure_level = logging.getLogger("azure").level
    original_azure_ai_level = logging.getLogger("azure.ai.ml").level
    original_azure_core_level = logging.getLogger("azure.core").level
    original_azure_identity_level = logging.getLogger("azure.identity").level
    original_urllib3_level = logging.getLogger("urllib3").level

    try:
        # Suppress info/debug but allow warnings for cleaner output
        logging.getLogger("azure").setLevel(logging.WARNING)
        logging.getLogger("azure.ai.ml").setLevel(logging.WARNING)
        logging.getLogger("azure.core").setLevel(logging.WARNING)
        logging.getLogger("azure.identity").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        returned_job = ml_client.jobs.create_or_update(job)

    finally:
        # Restore original logging levels
        logging.getLogger("azure").setLevel(original_azure_level)
        logging.getLogger("azure.ai.ml").setLevel(original_azure_ai_level)
        logging.getLogger("azure.core").setLevel(original_azure_core_level)
        logging.getLogger("azure.identity").setLevel(original_azure_identity_level)
        logging.getLogger("urllib3").setLevel(original_urllib3_level)

    # Print results
    print(f"✅ {job_description} job '{returned_job.name}' submitted successfully!")
    print(f"🔗 Studio URL: {returned_job.studio_url}")

    if debug_mode:
        print("\n🐛 Debug mode enabled - VS Code and JupyterLab services available in Azure ML Studio")

    # Record job metadata locally
    try:
        record_job_submission(cfg, returned_job, job_type, debug_mode)
    except Exception as e:
        logger.warning(f"Failed to record job submission to CSV: {e}")

    return returned_job


def get_model_description(cfg: DictConfig) -> str:
    """Get a description of the model configuration"""
    return f"DINOv3 LoRA Hypernet (rank={cfg.model.lora_rank}, 768-dim)"


def record_job_submission(cfg: DictConfig, job_obj, job_type: str, debug_mode: bool, csv_path: str = "job_submissions.csv") -> None:
    """Append a record of the submitted job to a local CSV file.

    Columns stored:
        timestamp, job_name, job_id, job_type, model_description, experiment_name, debug_mode,
        compute, environment, use_radio_labels, lora_rank
    """
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    model_desc = get_model_description(cfg)
    job_name = getattr(job_obj, "name", "<unknown>")
    job_id = getattr(job_obj, "id", job_name)
    experiment_name = cfg.azureml.experiment_name
    compute = cfg.azureml.compute_name
    environment = cfg.azureml.environment_name
    use_radio_labels = getattr(cfg.model, "use_radio_labels", False)
    lora_rank = getattr(cfg.model, "lora_rank", None)

    file_path = Path(csv_path)
    file_exists = file_path.exists()

    headers = [
        "timestamp",
        "job_name",
        "job_id",
        "job_type",
        "model_description",
        "experiment_name",
        "debug_mode",
        "compute",
        "environment",
        "use_radio_labels",
        "lora_rank",
    ]
    row = [
        timestamp,
        job_name,
        job_id,
        job_type,
        model_desc,
        experiment_name,
        str(debug_mode),
        compute,
        environment,
        str(use_radio_labels),
        str(lora_rank),
    ]

    with file_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    logger.info(f"🗂️ Recorded job submission to {file_path}")

# ======================== HYDRA APP ========================


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main submission function with Hydra configuration"""

    # Override debug_mode from command line if passed
    debug_mode = cfg.get("debug_mode", False)

    # Print job summary
    print("\n" + "=" * 60)
    print("🚀 CT2ECHO AZURE ML JOB SUBMISSION (HYDRA)")
    print("=" * 60)
    print(f"📝 Job Type: {cfg.job_type}")
    print(f"⚙️  Model: {get_model_description(cfg)}")
    print(f"🐛 Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")
    print(f"💻 Compute: {cfg.azureml.compute_name}")
    print(f"🏗️  Environment: {cfg.azureml.environment_name}")

    # Show Azure ML paths
    print("\n📂 Azure ML Data Paths:")
    print(f"  Pretrain weights: {cfg.azureml.data_base_path}/{cfg.azureml.pretrain_weight_path}")
    print(f"  Train/Val data: {cfg.azureml.data_base_path}/{cfg.azureml.train_val_path}")
    print(f"  Task embeddings: {cfg.azureml.data_base_path}/{cfg.azureml.task_embed_path}")
    print(f"  DINOv3 weights: {cfg.azureml.data_base_path}/{cfg.azureml.dinov3_path}")

    # Show configuration
    print("\n🔧 Hydra Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Show the command that will be sent to Azure ML
    print("\n📋 Azure ML Command Preview:")
    hydra_overrides = create_hydra_overrides_command(cfg)
    print(f"   python scripts/mtl_main_hydra.py {hydra_overrides}")

    print("=" * 60 + "\n")

    if cfg.dry_run:
        print("✅ DRY RUN MODE - Configuration validated, not submitting job")
        return

    # Submit job
    try:
        submit_job(cfg, cfg.job_type, debug_mode, cfg.get("display_name", None))
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
