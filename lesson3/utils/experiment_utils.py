import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json
import os
import logging
import numpy as np

from .model_utils import FullyConnectedModel, count_parameters
from .visualization_utils import plot_training_history, plot_weights_hist


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def run_epoch(model, data_loader, criterion, optimizer, is_training):
    model.train(is_training)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    data_iterator = tqdm(data_loader, desc=f"{'Training' if is_training else 'Testing'}")

    with torch.set_grad_enabled(is_training):
        for inputs, targets in data_iterator:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

            data_iterator.set_postfix(loss=total_loss / total_samples, acc=correct_predictions.double().item() / total_samples)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc.item()


def run_experiment(config, train_loader, test_loader, epochs=10, learning_rate=0.001, weight_decay=0.0):
    """Обучает одну сеть и сохраняет результат.

    Параметры
    ----------
    config : dict
        Описание модели и эксперимента.
    train_loader, test_loader : DataLoader
        Даные для обучения и проверки.
    epochs : int
        Количество эпох.
    learning_rate, weight_decay : float
        Гиперпараметры
    """
    logger.info(f"Начало эксперимента: {config['name']}")

    model = FullyConnectedModel(
        input_size=config['input_size'],
        num_classes=config['num_classes'],
        layer_config=config['layers']
    ).to(device)

    logger.info(f"Параметров в модели: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = {
        'train_losses': [], 'train_accs': [],
        'test_losses': [], 'test_accs': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        logger.info(f"Эпоха {epoch + 1}/{epochs}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, is_training=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, is_training=False)

        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    logger.info(f"Время обучения: {training_time:.2f} с")

    results_dir = os.path.join("results", config['experiment_type'])
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{config['name']}_results.json")

    results_data = {
        'config': config,
        'history': history,
        'training_time': training_time,
        'num_parameters': count_parameters(model),
        'final_test_accuracy': history['test_accs'][-1]
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=4)

    plot_path = os.path.join("plots", config['experiment_type'], f"{config['name']}_learning_curves.png")
    plot_training_history(history, f"Графики обучения: {config['name']}", save_path=plot_path, show=False)

    # Гистограмма весов модели
    weight_values = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    weights_path = os.path.join("plots", config['experiment_type'], f"{config['name']}_weights_hist.png")
    plot_weights_hist(weight_values, f"Распределение весов: {config['name']}", save_path=weights_path, show=False)

    logger.info(f"Эксперимент завершён: {config['name']}")
    return results_data