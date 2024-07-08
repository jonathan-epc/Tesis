import os
from timeit import default_timer
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from loguru import logger
import numpy as np
from time import sleep


def compute_metrics():
    mse, mae, r2 = np.random.rand(3)
    return mse, mae, r2

def cross_validate():
    k_folds = 5
    with tqdm(total=k_folds, desc=f"{k_folds} folds Cross Validation") as pbar:
        for fold, (train_idx, val_idx) in tqdm(enumerate(np.random.rand(5,2)), total=k_folds):
            fold_results = []
            logger.info(f"Starting fold {fold+1}/{k_folds}")
            logger.info(f"Training model for fold {fold+1}")
            train_model()
            val_loss, val_mse, val_mae, val_r2 = validate_model()
            fold_results.append((val_loss, val_mse, val_mae, val_r2))
            logger.log("METRIC", f"Fold {fold+1} results: Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}")
            pbar.update(1)
    
        avg_results = np.mean(fold_results, axis=0)
        logger.log("METRIC", f"Cross-validation results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}")
        return avg_results


def validate_model():
    val_loss, val_mse, val_mae, val_r2 = np.random.rand(4)
    return val_loss, val_mse, val_mae, val_r2

def train_model():
    num_epochs = 256
    with tqdm(total=num_epochs, desc = "Fold") as pbar:
        for epoch in range(num_epochs):
            train_loss, train_mse, train_mae, train_r2 = np.random.rand(4)
            val_loss, val_mse, val_mae, val_r2 = validate_model()
            sleep(0.005)
            if np.random.rand(1) >= 0.995:
                logger.info("Early stopping triggered")
                break
            pbar.update(1)
            pbar.set_postfix({"epoch":epoch, "Loss":train_loss, "MSE":train_mse, "MAE":train_mae, "R2":train_r2})

def test_model():
    test_loss, test_mse, test_mae, test_r2 = np.random.rand(4)
    logger.log("METRIC", f"Test Loss: {test_loss:.4f}\nTest MSE: {test_mse:.4f}\nTest MAE: {test_mae:.4f}\nTest R2: {test_r2:.4f}")


def cross_validation_procedure():
    logger.info("Starting cross-validation procedure")
    logger.info("Loading dataset")
    logger.info(f"Dataset split into training/validation ({256} samples) and test ({128} samples)")
    logger.info("Starting cross-validation")
    avg_results = cross_validate()

    logger.log("METRIC", f"Cross-validation completed.\n Avg results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}"    )
    logger.info("Testing model on the test dataset")
    logger.success("Cross-validation procedure completed")


def main():
    cross_validation_procedure()


if __name__ == "__main__":
    logger.remove()
    logger.add(
        "test.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {message}",
        rotation="10 MB",
        compression="zip",
        mode="a",
    )
    logger.level("METRIC", no=15, color="<white>")
    #logger.add(lambda msg: tqdm.write(msg, end=""))
    main()
