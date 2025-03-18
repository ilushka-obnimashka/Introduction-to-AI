from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


def load_data(data_dir: str, batch_size: int, train_transform, val_transform) -> \
        tuple[DataLoader, DataLoader, DataLoader, str]:
    """
    Загружает датасет, применяет аугментации "на лету", разделяет датасет на тренировочную, тестовую
    и валидационные части.

    Returns:
        train_loader, test_loader, val_loader, class_names
    """

    test_transform = val_transform

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

    class_names = sorted(dataset.classes)
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    dataset.targets = [class_to_idx[dataset.classes[target]] for target in
                       dataset.targets]
    dataset.class_to_idx = class_to_idx
    dataset.classes = class_names

    labels = [dataset.targets[i] for i in range(len(dataset))]

    train_idx, val_test_idx = train_test_split(range(len(dataset)), test_size=0.3, stratify=labels)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, stratify=[labels[i] for i in val_test_idx])

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(val_dataset, val_idx)
    test_dataset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader, val_loader, class_names
