# ML/scripts/run_transfer_learning.py


from pathlib import Path

import torch
from nconfig import get_config

from common.utils import set_seed, setup_logger
from ML.core.finetuner import (
    GEOM_NAMES,
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ModelFineTuner,
)

project_root = Path(__file__).resolve().parents[2]


def main():
    """
    Orchestrates the process of fine-tuning all pre-trained models on all other geometries.
    """
    logger = setup_logger()
    logger.info("--- Starting All Transfer Learning Jobs ---")

    try:
        base_config_template = get_config("nconfig.yml")
        set_seed(base_config_template.seed)
    except (FileNotFoundError, ValueError) as e:
        logger.error(
            f"CRITICAL: Failed to load base configuration. Aborting. Error: {e}"
        )
        return

    jobs_attempted = 0
    jobs_succeeded = 0

    for model_key, model_info in TRAINED_MODELS_INFO.items():
        source_geom_char = model_info["source_geom"]

        for target_geom_char, target_data_file_suffix in GEOMETRY_FILES.items():
            if target_geom_char == source_geom_char:
                continue  # Skip fine-tuning on the original geometry

            jobs_attempted += 1
            target_data_file = str(project_root / target_data_file_suffix)

            logger.info(
                f"\n--- Starting Job #{jobs_attempted}: Fine-tune {model_key.upper()} on {GEOM_NAMES[target_geom_char].upper()} ---"
            )

            # Construct a descriptive run name
            base_model_name = (
                f"{model_info['study_name']}_trial{model_info['trial_number']}"
            )
            finetune_run_name = f"ft_{model_key}_from_{base_model_name}_on_{GEOM_NAMES[target_geom_char]}"

            # Create a deep copy of the base config for this specific job
            current_job_config = base_config_template.copy(deep=True)

            # OPTIONAL: You can override fine-tuning parameters here if desired
            # e.g., use a smaller learning rate for fine-tuning
            # current_job_config.training.learning_rate = 1e-5

            try:
                fine_tuner = ModelFineTuner(
                    config=current_job_config,
                    source_model_info=model_info,
                    target_dataset_path=target_data_file,
                    finetune_run_name=finetune_run_name,
                )
                result = fine_tuner.finetune()

                if result is not None:
                    logger.info(
                        f"SUCCESS: Job '{finetune_run_name}' completed. Avg CV Loss: {result:.6f}"
                    )
                    jobs_succeeded += 1
                else:
                    logger.error(
                        f"FAILURE: Job '{finetune_run_name}' returned None (likely failed)."
                    )

            except Exception as e:
                logger.exception(
                    f"CRITICAL ERROR during job '{finetune_run_name}': {e}"
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info("\n--- All Transfer Learning Jobs Processed ---")
    logger.info(f"Total jobs attempted: {jobs_attempted}")
    logger.info(f"Total jobs succeeded: {jobs_succeeded}")


if __name__ == "__main__":
    main()
