# src/ML/scripts/generate_plots.py

import argparse
from pathlib import Path

from tqdm import tqdm

from common.utils import setup_logger
from ML.core.results import (
    GEOM_NAMES,
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ResultsLoader,
)
from ML.modules.metrics import evaluate_predictions
from ML.modules.plots import (
    PlotManager,
    plot_error_analysis,
    plot_field_comparison,
    plot_scatter_predictions,
)
from ML.plotting_style import set_plotting_style

project_root = Path(__file__).resolve().parents[2]
logger = setup_logger()


def generate_plots_for_trial(
    study_name: str,
    trial_number: int,
    eval_dataset_key: str,
    language: str,
):
    """
    Loads a specific trained model, evaluates it on a specific dataset, and generates a full suite of plots.
    """
    try:
        model_name_key = next(
            key
            for key, val in TRAINED_MODELS_INFO.items()
            if val["study_name"] == study_name
        )
        logger.info(
            f"--- Generating plots for Model '{model_name_key.upper()}' on Data '{GEOM_NAMES[eval_dataset_key]}' (Lang: {language}) ---"
        )

        results_loader = ResultsLoader(study_name=study_name, trial_number=trial_number)
        model_config = results_loader.config
        data_dir = Path(model_config.paths.data_dir)
        eval_dataset_filename = GEOMETRY_FILES[eval_dataset_key]
        eval_dataset_path = str(data_dir / eval_dataset_filename)
        logger.info(
            f"Evaluating model from {study_name} on dataset: {eval_dataset_path}"
        )
        eval_config = model_config.model_copy(deep=True)
        eval_config.data.file_path = eval_dataset_path
        predictions, targets = results_loader.run_inference_on_dataset(eval_config)

        metrics, per_case_df = evaluate_predictions(
            predictions,
            targets,
            model_config.data.output_fields,
            model_config.data.output_scalars,
        )

        plot_dir_name = (
            f"{language}/model_{model_name_key}_on_data_{GEOM_NAMES[eval_dataset_key]}"
        )
        plot_manager = PlotManager(base_dir=str(project_root / "plots" / plot_dir_name))
        title_prefix = (
            f"Model: {model_name_key.upper()} | Data: {GEOM_NAMES[eval_dataset_key]}"
        )

        plot_scatter_predictions(
            predictions,
            targets,
            model_config.data.output_fields,
            model_config.data.output_scalars,
            plot_manager,
            metrics,
            title_prefix,
            language,
        )
        plot_error_analysis(
            predictions,
            targets,
            model_config.data.output_fields,
            plot_manager,
            title_prefix,
            language,
        )

        # --- Generate field comparisons for ALL output fields ---
        if not per_case_df.empty and model_config.data.output_fields:
            # Loop through each output field variable
            for field_idx, field_name in enumerate(model_config.data.output_fields):
                field_df = per_case_df[per_case_df["variable"] == field_name].copy()

                if not field_df.empty:
                    best_case_row = field_df.loc[field_df["rmse"].idxmin()]
                    best_case_id = int(best_case_row["case_id"])

                    worst_case_row = field_df.loc[field_df["rmse"].idxmax()]
                    worst_case_id = int(worst_case_row["case_id"])

                    median_case_row = field_df.loc[
                        (field_df["rmse"] - field_df["rmse"].median()).abs().idxmin()
                    ]
                    median_case_id = int(median_case_row["case_id"])
                    cases_to_plot = {
                        "best": best_case_id,
                        "worst": worst_case_id,
                        "median": median_case_id,
                    }

                    logger.debug(
                        f"Plotting comparison cases for variable '{field_name}': {cases_to_plot}"
                    )

                    for name, case_id in cases_to_plot.items():
                        plot_field_comparison(
                            prediction=predictions[0][case_id, field_idx],
                            target=targets[0][case_id, field_idx],
                            variable_name=f"{field_name} {name}",  # e.g., "H best"
                            plot_manager=plot_manager,
                            case_id=case_id,
                            title_prefix=title_prefix,
                            language=language,
                        )
        logger.info(f"Plots saved in: {plot_manager.base_dir}")
        return True

    except Exception as e:
        logger.exception(
            f"Failed plots for model {study_name} on data {eval_dataset_key}: {e}"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation plots."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=f"Specific model to plot (e.g., ddb, idb), or 'all'. Choices: {list(TRAINED_MODELS_INFO.keys())}",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="all",
        help=f"Specific dataset to evaluate on (e.g., b, s, n), or 'all'. Choices: {list(GEOMETRY_FILES.keys())}",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["en", "es", "all"],
        help="Language for plots.",
    )
    args = parser.parse_args()

    set_plotting_style()
    models_to_plot = (
        list(TRAINED_MODELS_INFO.items())
        if args.model == "all"
        else [(args.model, TRAINED_MODELS_INFO[args.model])]
    )
    datasets_to_eval = (
        list(GEOMETRY_FILES.keys()) if args.data == "all" else [args.data]
    )
    languages = ["en", "es"] if args.lang == "all" else [args.lang]

    success_count = 0
    total_jobs = len(models_to_plot) * len(datasets_to_eval) * len(languages)

    pbar = tqdm(total=total_jobs, desc="Overall Progress")

    for model_key, model_info in models_to_plot:
        for data_key in datasets_to_eval:
            for lang in languages:
                # Update the progress bar description for the current job
                pbar.set_description(
                    f"Processing Model '{model_key.upper()}' on Data '{data_key.upper()}' ({lang})"
                )

                logger.info("=" * 80)
                if generate_plots_for_trial(
                    model_info["study_name"], model_info["trial_number"], data_key, lang
                ):
                    success_count += 1

                # Update the progress bar
                pbar.update(1)

    pbar.close()  # Close the progress bar

    logger.info("--- Plot Generation Summary ---")
    logger.info(
        f"Successfully completed {success_count} out of {total_jobs} plotting jobs."
    )


if __name__ == "__main__":
    main()
