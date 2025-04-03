import os
from datetime import datetime

import torch
from torch import optim, nn
from torchvision import models
from torchvision.transforms import v2

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

DATA_DIR = "my_simpsons"
NUM_CLASSES = 42
BATCH_SIZE = 128
IMG_SIZE = (224, 224)

MODEL_NAME = "MobileNetV2"
MODEL = models.mobilenet_v2(pretrained=False)
MODEL.classifier[1] = nn.Linear(1280, NUM_CLASSES)
#MODEL.classifier[6] = nn.Linear(4096, NUM_CLASSES)
#MODEL.fc = nn.Linear(MODEL.fc.in_features, NUM_CLASSES)
MODEL.to(DEVICE)

NUM_EPOCHS = 30
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001

OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
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

INFERENCE_TRANSFORM = transform = v2.Compose([
    v2.Resize(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRAIN_TRANSFORM = v2.Compose([
        v2.Resize(IMG_SIZE),
        v2.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
                                        scale=(0.9, 1.1))], p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomPerspective(distortion_scale=0.4, p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])