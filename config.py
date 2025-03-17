import os
from datetime import datetime

import torch
from torch import optim, nn
from torchvision import models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "my_simpsons"
NUM_CLASSES = 42
BATCH_SIZE = 64
IMG_SIZE = (224, 224)

MODEL_NAME = "resnet34"
MODEL = models.resnet34(pretrained=False)
MODEL.fc = nn.Linear(MODEL.fc.in_features, NUM_CLASSES)
MODEL.to(DEVICE)

NUM_EPOCHS = 30
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.001

OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
CRITERION = nn.CrossEntropyLoss()


def get_save_dir(model_name, optimizer, num_epochs, lr):
    """
    Создает имя директории на основе параметров модели, обучения и текущего времени (без года).
    """
    optimizer_name = optimizer.__class__.__name__
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    dir_name = f"{model_name}_{optimizer_name}_epochs{num_epochs}_lr{lr}_{current_time}"
    return dir_name


SAVE_DIR = os.path.join("saved_models",
                        get_save_dir(MODEL_NAME, OPTIMIZER, NUM_EPOCHS,
                                     LEARNING_RATE))
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_model.pth")
GRAPHS_DIR = os.path.join(SAVE_DIR, "graphs")
