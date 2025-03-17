import torch
from tqdm import tqdm

from .graph_utils import save_train_val_graphs


def train(
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        num_epochs: int,
        train_loader,
        val_loader,
        model_save_path: str,
        graphs_dir: str) -> None:
    """
    Обучает модель на тренировочных данных и оценивает ее на валидационных.
    """

    best_val_accuracy = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = []
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss.append(loss.item())

        mean_train_loss = sum(running_train_loss) / len(running_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(mean_train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        running_val_loss = []
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(mean_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"""Epoch [{epoch + 1}/{num_epochs}], 
        Train loss: {mean_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%
        Val Loss: {mean_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%""")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model, model_save_path)

    save_train_val_graphs(train_losses, val_losses, train_accuracies, val_accuracies, graphs_dir)
    print("\033[91mTraining complete\033[0m")
