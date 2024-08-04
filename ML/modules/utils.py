import numpy as np
import torch
from loguru import logger


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, save_path="savepoints/best_model.pth"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.epoch = 0
        self.save_path = save_path

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        self.epoch = epoch

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f"New best model saved at epoch {self.epoch+1}")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)