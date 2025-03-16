from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split


def load_data(img_size: tuple[int, int], data_dir: str, batch_size: int) -> \
tuple[DataLoader, DataLoader, DataLoader, str]:
    """
    Загружает датасет, применяет аугментации "на лету", разделяет датасет на тренировочную, тестовую
    и валидационные части.

    Returns:
        train_loader, test_loader, val_loader, class_names
    """
    
    transform = v2.Compose([
        v2.Resize(img_size),
        v2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
                                        scale=(0.9, 1.1))], p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomPerspective(distortion_scale=0.4, p=0.5), 
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    class_names = sorted(dataset.classes)

    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    dataset.targets = [class_to_idx[dataset.classes[target]] for target in
                       dataset.targets]
    dataset.class_to_idx = class_to_idx
    dataset.classes = class_names

    labels = [dataset.targets[i] for i in range(len(dataset))]

    train_idx, val_test_idx = train_test_split(range(len(dataset)),
                                               test_size=0.3, stratify=labels)

    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.7,
                                         stratify=[labels[i] for i in val_test_idx])

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader, val_loader, class_names
