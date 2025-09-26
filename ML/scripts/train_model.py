# ML/scripts/train_model.py

import sys

from common.utils import set_seed, setup_logger
from ML.core.trainer import ModelTrainer
from nconfig import get_config


def main():
    """
    Main function to execute a single model training run.
    """
    logger = setup_logger()
    try:
        # Configuration for the run is loaded from the single nconfig.yml file
        config = get_config("nconfig.yml")
        set_seed(config.seed)

        logger.info(f"Starting training for model: {config.model.name}")
        logger.info(f"Dataset: {config.data.file_path}")

        trainer = ModelTrainer(config)
        avg_cv_loss = trainer.train()

        if avg_cv_loss is not None:
            logger.info(f"Training run completed. Average CV loss: {avg_cv_loss:.6f}")
        else:
            logger.error("Single training run failed.")
        logger.info("--- Training Run Finished ---")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration or setup error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
