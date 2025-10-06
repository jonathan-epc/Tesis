# src/ML/scripts/generate_plots.py

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Correct, absolute imports for the src-layout
from common.utils import setup_logger
from ML.core.results import (
    GEOM_NAMES,
    GEOMETRY_FILES,
    TRAINED_MODELS_INFO,
    ResultsLoader,
)
from ML.modules.data import HDF5Dataset
from ML.modules.metrics import evaluate_predictions
from ML.modules.plots import (
    PlotManager,
    plot_error_analysis,
    plot_field_comparison,
    plot_scatter_predictions,
)
from ML.plotting_style import set_plotting_style

# Define project root relative to this file's location. This is robust.
project_root = Path(__file__).resolve().parents[3]
logger = setup_logger()


def run_single_job(model_key: str, data_key: str, lang: str):
    """
    This is the core "worker" function. It runs ONE evaluation and plotting job
    and then exits. It is designed to be called in a completely isolated process.
    """
    try:
        dataset_name = GEOM_NAMES[data_key]
        logger.info("=" * 80)
        logger.info(
            f"STARTING JOB: Model '{model_key.upper()}' on Data '{dataset_name}' (Lang: {lang})"
        )

        # 1. Load the model and configuration
        results_loader = ResultsLoader(
            study_name=TRAINED_MODELS_INFO[model_key]["study_name"],
            trial_number=TRAINED_MODELS_INFO[model_key]["trial_number"],
        )
        model = results_loader.model
        config = results_loader.config
        hparams = results_loader.hparams

        # 2. Prepare the evaluation dataset (NO PRELOADING)
        eval_config = config.model_copy(deep=True)
        data_dir = project_root / eval_config.paths.data_dir
        eval_config.data.file_path = str(data_dir / GEOMETRY_FILES[data_key])
        eval_config.data.preload_hdf5 = False  # Explicitly disable preloading

        eval_dataset = HDF5Dataset.from_config(
            eval_config, file_path=eval_config.data.file_path
        )

        test_size = int(eval_config.training.test_frac * len(eval_dataset))
        train_val_size = len(eval_dataset) - test_size
        _, test_dataset = torch.utils.data.random_split(
            eval_dataset,
            [train_val_size, test_size],
            generator=torch.Generator().manual_seed(eval_config.seed),
        )

        loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=hparams.get("batch_size", 32), shuffle=False
        )

        # 3. Run inference
        all_field_preds, all_scalar_preds = [], []
        all_field_targs, all_scalar_targs = [], []

        with torch.no_grad():
            for batch in loader:
                inputs, targets_batch = batch
                inputs_on_device = (
                    [t.to(config.device) for t in inputs[0]],
                    [t.to(config.device) for t in inputs[1]],
                )
                preds = model(inputs_on_device)

                if hparams.get("normalize_output", False):
                    from ML.modules.utils import denormalize_outputs_and_targets

                    preds, targets_batch = denormalize_outputs_and_targets(
                        preds, targets_batch, eval_dataset, eval_config, True
                    )

                if preds[0] is not None:
                    all_field_preds.append(preds[0].cpu())
                if preds[1] is not None:
                    all_scalar_preds.append(preds[1].cpu())
                if targets_batch[0]:
                    all_field_targs.append(torch.stack(targets_batch[0], dim=1).cpu())
                if targets_batch[1]:
                    all_scalar_targs.append(torch.stack(targets_batch[1], dim=1).cpu())

        predictions = (
            torch.cat(all_field_preds) if all_field_preds else None,
            torch.cat(all_scalar_preds) if all_scalar_preds else None,
        )
        targets = (
            torch.cat(all_field_targs) if all_field_targs else None,
            torch.cat(all_scalar_targs) if all_scalar_targs else None,
        )

        # 4. Calculate Metrics & Setup Plotting
        metrics, per_case_df = evaluate_predictions(
            predictions, targets, config.data.output_fields, config.data.output_scalars
        )
        plot_dir_name = f"{lang}/model_{model_key}_on_data_{dataset_name}"
        plot_manager = PlotManager(base_dir=str(project_root / "plots" / plot_dir_name))
        title_prefix = f"Model: {model_key.upper()} | Data: {dataset_name}"

        # 5. Generate Plots
        plot_scatter_predictions(
            predictions,
            targets,
            config.data.output_fields,
            config.data.output_scalars,
            plot_manager,
            metrics,
            title_prefix,
            lang,
        )
        plot_error_analysis(
            predictions,
            targets,
            config.data.output_fields,
            plot_manager,
            title_prefix,
            lang,
        )

        if not per_case_df.empty and config.data.output_fields:
            for field_idx, field_name in enumerate(config.data.output_fields):
                field_df = per_case_df[per_case_df["variable"] == field_name]
                if not field_df.empty:
                    cases_to_plot = {
                        "best": int(field_df.loc[field_df["rmse"].idxmin()]["case_id"]),
                        "worst": int(
                            field_df.loc[field_df["rmse"].idxmax()]["case_id"]
                        ),
                        "median": int(
                            field_df.loc[
                                (field_df["rmse"] - field_df["rmse"].median())
                                .abs()
                                .idxmin()
                            ]["case_id"]
                        ),
                    }
                    for name, case_id in cases_to_plot.items():
                        plot_field_comparison(
                            prediction=predictions[0][case_id, field_idx],
                            target=targets[0][case_id, field_idx],
                            variable_name=f"{field_name} ({name})",
                            plot_manager=plot_manager,
                            case_id=case_id,
                            title_prefix=title_prefix,
                            language=lang,
                        )

        logger.info(
            f"SUCCESS: Plots for '{model_key}' on '{dataset_name}' saved in: {plot_manager.base_dir}"
        )

    except Exception as e:
        logger.exception(
            f"FATAL ERROR in worker for job ({model_key}, {data_key}, {lang}): {e}"
        )
        # Exit with a non-zero status code to signal failure to the manager process
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation plots."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model key to plot (e.g., 'ddb', or 'all').",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="all",
        help="Dataset key to evaluate on (e.g., 'b', or 'all').",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language for plots (e.g., 'en', or 'all').",
    )
    args = parser.parse_args()

    # --- SCRIPT BEHAVIOR SWITCH ---
    # If specific arguments are given, run as a WORKER.
    # Otherwise, run as the MANAGER.

    is_worker = args.model != "all" and args.data != "all" and args.lang != "all"

    if is_worker:
        # --- WORKER LOGIC ---
        # This code runs inside the separate subprocess.
        set_plotting_style()
        run_single_job(args.model, args.data, args.lang)

    else:
        # --- MANAGER LOGIC ---
        # This code runs only once to orchestrate the subprocesses.
        logger.info(
            "--- Running as MANAGER: Spawning worker processes for each job ---"
        )

        models_to_plot = (
            list(TRAINED_MODELS_INFO.keys()) if args.model == "all" else [args.model]
        )
        datasets_to_eval = (
            list(GEOMETRY_FILES.keys()) if args.data == "all" else [args.data]
        )
        languages = ["en", "es"] if args.lang == "all" else [args.lang]

        all_jobs = []
        for model_key in models_to_plot:
            for data_key in datasets_to_eval:
                for lang in languages:
                    all_jobs.append((model_key, data_key, lang))

        success_count = 0
        with tqdm(total=len(all_jobs), desc="Overall Plotting Progress") as pbar:
            for model_key, data_key, lang in all_jobs:
                pbar.set_description(
                    f"Job: {model_key.upper()} on {GEOM_NAMES[data_key]} ({lang})"
                )

                # Construct the command to call this script on itself
                command = [
                    sys.executable,  # The path to the current Python interpreter
                    __file__,  # The full path to this script
                    "--model",
                    model_key,
                    "--data",
                    data_key,
                    "--lang",
                    lang,
                ]

                # Run the command in a separate process.
                # stdout and stderr are captured to prevent cluttering the main progress bar.
                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode == 0:
                    success_count += 1
                else:
                    # If a worker fails, log its output for debugging
                    logger.error(
                        f"--- Worker FAILED for job ({model_key}, {data_key}, {lang}) ---"
                    )
                    logger.error("--- STDOUT ---:\n" + result.stdout)
                    logger.error("--- STDERR ---:\n" + result.stderr)

                pbar.update(1)

        logger.info("--- Plot Generation Summary ---")
        logger.info(
            f"Successfully completed {success_count} out of {len(all_jobs)} plotting jobs."
        )


if __name__ == "__main__":
    main()
