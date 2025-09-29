# ML/scripts/run_hyperparameter_search.py

import sys

from common.utils import set_seed, setup_logger
from nconfig import get_config

from ML.core.optimizer import HyperparameterOptimizer


def main():
    """
    Main function to execute the hyperparameter optimization process.
    """
    logger = setup_logger()
    try:
        config = get_config("nconfig.yml")
        set_seed(config.seed)

        logger.info(
            f"Starting hyperparameter search for study: {config.optuna.study_name}"
        )
        optimizer = HyperparameterOptimizer(config)
        results = optimizer.run_optimization()

        if results:
            logger.info(
                f"Optimization completed. Best CV value: {results.best_value:.6f}"
            )
        else:
            logger.warning("Optimization process did not yield results.")
        logger.info("--- Hyperparameter Search Finished ---")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration or setup error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
