import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from tqdm import tqdm

from .graph_utils import save_test_graphs


def test(
        model_best_path: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: list[str],
        test_loader: DataLoader,
        save_dir: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Тестирует модель на тестовом наборе данных и вычисляет метрики для каждого класса.
    Returns:
        precision (np.ndarray) - Массив precision для каждого класса.
        mi
        recall (np.ndarray): Массив recall для каждого класса.

        f1 (np.ndarray): Массив F1-score для каждого класса.

        accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)

        confusion_matrix (np.ndarray): Матрица ошибок (confusion matrix).
    """

    model = torch.load(model_best_path, map_location=device, weights_only=False)

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average=None).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=None).to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average=None).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=None).to(device)

    micro_accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    micro_precision_metric = MulticlassPrecision(num_classes=num_classes, average="micro").to(device)
    micro_recall_metric = MulticlassRecall(num_classes=num_classes, average="micro").to(device)
    micro_f1_metric = MulticlassF1Score(num_classes=num_classes, average="micro").to(device)

    model.eval()

    labels_true = []
    labels_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test", unit="batch", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

            micro_accuracy_metric.update(predicted, labels)
            micro_precision_metric.update(predicted, labels)
            micro_recall_metric.update(predicted, labels)
            micro_f1_metric.update(predicted, labels)

            labels_true.extend(labels.cpu().numpy())
            labels_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_metric.compute().cpu().numpy()
    precision = precision_metric.compute().cpu().numpy()
    recall = recall_metric.compute().cpu().numpy()
    f1 = f1_metric.compute().cpu().numpy()

    micro_accuracy = micro_accuracy_metric.compute().item()
    micro_precision = micro_precision_metric.compute().item()
    micro_recall = micro_recall_metric.compute().item()
    micro_f1 = micro_f1_metric.compute().item()

    model_confusion_matrix = confusion_matrix(labels_true, labels_pred)

    save_test_graphs(accuracy, micro_accuracy, precision, micro_precision, recall, micro_recall, f1, micro_f1, model_confusion_matrix, save_dir, class_names)

    print("\033[91mTesting complete\033[0m")

    return accuracy, precision, recall, model_confusion_matrix
