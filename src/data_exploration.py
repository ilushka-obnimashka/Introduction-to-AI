import csv
import os

import matplotlib.pyplot as plt


def analyze_class_distribution(dataset_path: str) -> dict[str, int]:
    """
    Формирует словарь, содержащий имя класса и количество его экземпляров в датасете.
    Parameters:
        dataset_path (str): путь к датасете для анализа
    Returns:
        dict[str, int]: словарь, где ключом является имя класса, а значением — количество его экземпляров
    """
    class_counts = {}
    for classname in os.listdir(dataset_path):
        class_counts[classname] = len(os.listdir(os.path.join(dataset_path, classname)))
    return class_counts


def plot_class_distribution(class_counts: dict[str, int], dataset_path: str) -> None:
    """
    Строит частотную гистограмму распределения классов, сохраняет в текущую папку.
    Parameters:
        class_counts: словарь, где ключом является имя класса, а значением — количество его экземпляров
        dataset_path: путь к корневой директории датасета, содержащей поддиректории с изображениями для каждого класса
    Returns:
         None
    """

    dataset_name = os.path.basename(dataset_path)
    filename = dataset_name + "_class_distribution.png"

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))
    plt.bar(classes, counts)
    plt.title("Distribution of classes")
    plt.xlabel("classes")
    plt.ylabel("Amount of images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)


def save_class_distributio(class_counts: dict[str, int], dataset_path: str) -> None:
    """
    Сохраняет информацию о балансе классов датасета в CSV-файл.
    Parameters:
        class_counts: словарь, где ключом является имя класса, а значением — количество его экземпляров.
        dataset_path: путь к корневой директории датасета, содержащей поддиректории с изображениями для каждого класса
    Returns:
         None
    """

    dataset_name = os.path.basename(dataset_path)
    filename = dataset_name + "_class_distribution.csv"

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Class", "Items"]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for class_name, count in class_counts.items():
            writer.writerow({"Class": class_name, "Items": count})


def main() -> None:
    dataset_path = "../orig"

    class_count = analyze_class_distribution(dataset_path)
    plot_class_distribution(class_count, dataset_path)
    save_class_distributio(class_count, dataset_path)


if __name__ == '__main__':
    main()
