"""
config.py
=========
Global configuration. Two presets are exposed:

  QUICK  - small dims / few epochs / synthetic-friendly, finishes in
           well under a minute on CPU. This is what the dashboard runs
           by default so clicking "Run pipeline" always returns a result
           fast, regardless of hardware.
  FULL   - the original research-scale hyperparameters from the
           reference framework. Meant for a real GPU box with the real
           NASA CMAPSS data downloaded via kagglehub.
"""
import os
import random
import numpy as np
import torch

SELECTED_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21',
]
OP_SETTINGS = ['op_setting_1', 'op_setting_2', 'op_setting_3']
FEATURES = OP_SETTINGS + SELECTED_SENSORS
NUM_FEATURES = len(FEATURES)


class Config:
    def __init__(self, mode="quick"):
        self.mode = mode
        self.RUL_CAP = 125
        self.OP_SETTINGS = OP_SETTINGS
        self.SELECTED_SENSORS = SELECTED_SENSORS
        self.FEATURES = FEATURES
        self.NUM_FEATURES = NUM_FEATURES
        self.EVT_TAIL_QUANTILE = 0.90
        self.SEED = 42
        self.LAMBDA_ANOMALY = 1.0
        self.LAMBDA_FAULT = 1.0
        self.LAMBDA_RUL = 1.0
        self.LAMBDA_PHYSICS = 0.3
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mode == "full":
            self.SEQ_LEN = 30
            self.D_MODEL = 64
            self.N_HEADS = 4
            self.FF_DIM = 128
            self.N_TRANSFORMER_LAYERS = 2
            self.LATENT_DIM = 32
            self.GAT_HEADS = 4
            self.GAT_OUT_DIM = 8
            self.DROPOUT = 0.2
            self.MC_SAMPLES = 20
            self.LR = 1e-3
            self.PRETRAIN_EPOCHS = 10
            self.FINETUNE_EPOCHS = 30
            self.BATCH_SIZE = 128
            self.EVAL_BATCH_SIZE = 256
            self.N_ENGINES_SYNTH = 100
        else:  # quick demo preset
            self.SEQ_LEN = 16
            self.D_MODEL = 24
            self.N_HEADS = 2
            self.FF_DIM = 48
            self.N_TRANSFORMER_LAYERS = 1
            self.LATENT_DIM = 12
            self.GAT_HEADS = 2
            self.GAT_OUT_DIM = 6
            self.DROPOUT = 0.15
            self.MC_SAMPLES = 4
            self.LR = 2e-3
            self.PRETRAIN_EPOCHS = 2
            self.FINETUNE_EPOCHS = 3
            self.BATCH_SIZE = 64
            self.EVAL_BATCH_SIZE = 128
            self.N_ENGINES_SYNTH = 24

        self.DOMAIN = "FD001"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
