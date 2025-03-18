from config import *
from src.data_loader import load_data
from src.test import test
from src.train import train


def main():
    train_loader, test_loader, val_loader, class_names = load_data(DATA_DIR, BATCH_SIZE, TRAIN_TRANSFORM,
                                                                   INFERENCE_TRANSFORM)

    train(MODEL, DEVICE, OPTIMIZER, CRITERION, NUM_EPOCHS, train_loader,
          val_loader, MODEL_SAVE_PATH, GRAPHS_DIR)

    test(MODEL_SAVE_PATH, DEVICE, NUM_CLASSES, class_names, test_loader, GRAPHS_DIR)


if __name__ == '__main__':
    main()
