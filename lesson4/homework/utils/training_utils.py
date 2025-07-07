import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader, desc="Testing " if is_test else "Training")

    with torch.no_grad() if is_test else torch.enable_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)

            if not is_test and optimizer is not None:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            if not is_test and optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1), acc=f'{100. * correct / total:.2f}%')

    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': []
    }

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        print(
            f'Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

    return history

def get_predictions(model, data_loader, device='cpu'):
    """Получает предсказания модели для всего датасета."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_preds, all_targets