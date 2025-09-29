# ML/scripts/run_trial_repeat.py

import argparse
import sys

from common.utils import set_seed, setup_logger
from nconfig import get_config

from ML.core.repeater import TrialRepeater


def main():
    """
    Main function to execute the trial repetition process.

    This script loads the configuration, identifies a specific trial from an
    Optuna study, and re-runs the training process using that trial's exact
    hyperparameters.

    IMPORTANT: Before running, ensure that the `nconfig.yml` file is configured
    to use the same study (`optuna.study_name`) and dataset (`data.file_path`,
    `data.inputs`, `data.outputs`) that the original trial was run on.
    """
    parser = argparse.ArgumentParser(
        description="Re-run a specific Optuna trial with its saved hyperparameters.",
        formatter_class=argparse.RawTextHelpFormatter,  # To display the IMPORTANT note
    )
    parser.add_argument(
        "trial_id",
        type=int,
        help="The integer ID of the Optuna trial to repeat.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    try:
        config = get_config("nconfig.yml")
        set_seed(config.seed)

        logger.info(
            f"Attempting to repeat trial_id={args.trial_id} from study '{config.optuna.study_name}'"
        )

        repeater = TrialRepeater(config=config, trial_id=args.trial_id)
        result = repeater.run()

        if result is not None:
            logger.info(
                f"Successfully finished repeating trial {args.trial_id}. Final Avg CV Loss: {result:.6f}"
            )
        else:
            logger.error(f"Failed to repeat trial {args.trial_id}.")
        logger.info("--- Trial Repeat Finished ---")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration or setup error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the run: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
