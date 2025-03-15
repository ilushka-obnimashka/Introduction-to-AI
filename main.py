from src.data_loader import load_data
from src.train import train
from src.test import test
from config import *


def main():
    train_loader, test_loader, val_loader, class_names = load_data(IMG_SIZE,
                                                                   DATA_DIR,
                                                                   BATCH_SIZE)

    train(MODEL, DEVICE, OPTIMIZER, CRITERION, NUM_EPOCHS, train_loader,
          val_loader, MODEL_SAVE_PATH, GRAPHS_DIR)

    test(MODEL, DEVICE, NUM_CLASSES, class_names, test_loader, GRAPHS_DIR)


if __name__ == '__main__':
    main()
