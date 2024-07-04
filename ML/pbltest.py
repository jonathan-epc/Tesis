from loguru import logger
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, TaskProgressColumn, track
from rich.table import Table
from time import sleep
from tqdm.autonotebook import tqdm


class MetricsColumn(TaskProgressColumn):
    def render(self, task):
        return f"Loss={task.fields['val_loss']:.4f}, MSE={task.fields['val_mse']:.4f}, MAE={task.fields['val_mae']:.4f}, R2={task.fields['val_r2']:.4f}"


def create_progress_bar(transient=False, additional_columns=[]):
    base_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ] + additional_columns
    return Progress(*base_columns, transient=transient)


def train_model(total_epochs=256):
    for epoch in tqdm(range(total_epochs)):
        sleep(0.01)
        train_loss, train_mse, train_mae, train_r2 = np.random.rand(4)
        val_loss, val_mse, val_mae, val_r2 = np.random.rand(4)
        if np.random.rand(1) >= 0.995:
            logger.info("Early stopping triggered")
            break


def cross_validate(k_folds):
    fold_results = []

    for fold, task in tqdm(enumerate(range(5)), total = 5):
        logger.info(f"Starting fold {fold+1}/{k_folds}")
        logger.info(f"Training model for fold {fold+1}")
        train_model()
        val_loss, val_mse, val_mae, val_r2 = np.random.rand(4)

        fold_results.append((val_loss, val_mse, val_mae, val_r2))
        logger.info(
            f"Fold {fold+1} results: Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}"
        )

    avg_results = np.mean(fold_results, axis=0)
    logger.info(
        f"Cross-validation results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}"
    )
    return avg_results


def test_model():
    test_loss, test_mse, test_mae, test_r2 = np.random.rand(4)
    logger.info(
        f"Test Loss: {test_loss:.4f} Test MSE: {test_mse:.4f} Test MAE: {test_mae:.4f} Test R2={test_r2:.4f}"
    )


def cross_validation_procedure():
    logger.info("Starting cross-validation procedure")
    logger.info("Loading dataset")
    logger.info(f"Dataset split into training/validation ({256} samples) and test ({128} samples)")
    logger.info("Starting cross-validation")
    avg_results = cross_validate(k_folds=5)
    logger.info(
        f"Cross-validation completed. Avg results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}"
    )
    logger.info("Testing model on the test dataset")
    test_model()
    logger.info("Cross-validation procedure completed")


def main():
    cross_validation_procedure()


if __name__ == "__main__":
    console = Console()
    logger.remove()
    logger.add(RichHandler(console=console), format="{message}")
    logger.add(
        "test.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {message}",
        rotation="10 MB",
        compression="zip",
        mode="a"
    )
    main()


from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table

class MetricsColumn(TaskProgressColumn):
    def render(self, task):
        return f"[blue]Loss[/blue]={task.fields['val_loss']:.4f}, " \
               f"[green]MSE[/green]={task.fields['val_mse']:.4f}, " \
               f"[yellow]MAE[/yellow]={task.fields['val_mae']:.4f}, " \
               f"[red]R2[/red]={task.fields['val_r2']:.4f}"

def create_progress_bar(transient=True, additional_columns=[]):
    base_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ] + additional_columns
    return Progress(*base_columns, transient=transient)
k_folds = 5
overall_progress = create_progress_bar()
task_overall = overall_progress.add_task("Cross Validation", total=k_folds)
fold_progress = create_progress_bar(
    additional_columns=[SpinnerColumn(), MetricsColumn()]
)

progress_table = Table.grid()
progress_table.add_row(overall_progress)
progress_table.add_row(fold_progress)

console = Console()
with console:
    with overall_progress, fold_progress:
        for fold in range(k_folds):
            overall_progress.update(task_overall, advance=1)
            task_fold = fold_progress.add_task(f"Fold {fold+1}/{k_folds}", total=total_steps)
            for step in range(total_steps):
                # Simulate training step
                fold_progress.update(task_fold, advance=1, val_loss=0.01, val_mse=0.02, val_mae=0.03, val_r2=0.95)
                time.sleep(0.1)  # Simulate time taken for a step
            fold_progress.remove_task(task_fold)


import time
from rich.progress import track

for i in track(range(20), description="Processing..."):
    time.sleep(1)  # Simulate work being done

import time

from rich.progress import Progress

with Progress() as progress:

    task1 = progress.add_task("[red]Downloading...", total=1000)
    task2 = progress.add_task("[green]Processing...", total=1000)
    task3 = progress.add_task("[cyan]Cooking...", total=1000)

    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)

import time
from tqdm.rich import trange, tqdm


for i in trange(10):
    time.sleep(1)
