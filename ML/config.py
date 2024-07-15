# config.py
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 256
ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20
NUMPOINTS_X = 401
NUMPOINTS_Y = 11
VARIABLES = ["F", "H", "Q", "S", "U", "V"]
PARAMETERS = ["H0", "Q0", "SLOPE", "n"]