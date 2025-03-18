import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_train_val_graphs(train_losses, val_losses, train_accuracies,
                          val_accuracies, save_dir):
    """
    Сохраняет графики потерь и точности в виде изображений.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use("ggplot")
    plt.figure(figsize=(12, 6))
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()


def save_test_graphs(
        accuracy: np.ndarray[np.float64],
        micro_accuracy: np.float64,
        precision: np.ndarray[np.float64],
        micro_precision: np.float64,
        recall: np.ndarray[np.float64],
        micro_recall: np.float64,
        f1: np.ndarray[np.float64],
        micro_f1: np.float64,
        confusion_matrix: np.ndarray[np.float64],
        save_dir: str,
        class_names: list[str]
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    metrics_per_class = {"Accuracy": (accuracy, micro_accuracy),
                         "Recall": (recall, micro_recall),
                         "Precision": (precision, micro_precision),
                         "F1": (f1, micro_f1)}
    plt.style.use("ggplot")

    for (metricName, metric_vals) in metrics_per_class.items():
        metric_per_class, micro_metric = metric_vals

        plt.figure(figsize=(12, 6))
        plt.bar(class_names, metric_per_class,
                label=f"micro: {np.mean(micro_metric):.2f}")
        plt.title("Test_" + metricName + " per Class")
        plt.xlabel("Class")
        plt.ylabel(metricName)
        plt.xticks(rotation=90)
        plt.grid(axis="y")

        plt.legend(loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, metricName + ".png"))
        plt.close()

    plt.figure(figsize=(15, 12))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "Test_Confusion_matrix.png"))
    plt.close()
