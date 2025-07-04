import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

def plot_training_history(history, title, save_path=None, show: bool = False):
    """Отображает кривые обучения.
    одну для потерь, вторую — для точности. Это упрощает дальнейшую
    Параметры

    history : dict
        Словарь с ключами `train_losses`, `test_losses`, `train_accs` и
        `test_accs`
    title : str
        Заголовок, который будет добавлен к каждой фигуре.
    save_path : str | None
        Базовый путь до файла
    show : bool, default=False
        Отобразить ли на экране
    """

    # Извлекаем данные
    train_losses = history.get('train_losses', [])
    test_losses = history.get('test_losses', [])
    train_accs = history.get('train_accs', [])
    test_accs = history.get('test_accs', [])
    #Loss
    loss_fig, loss_ax = plt.subplots(figsize=(7, 5))
    loss_ax.plot(train_losses, label='Train Loss')
    loss_ax.plot(test_losses, label='Test Loss')
    loss_ax.set_title(f"{title} — Loss")
    loss_ax.set_xlabel('Epochs')
    loss_ax.set_ylabel('Loss')
    loss_ax.legend()
    loss_ax.grid(True)

    #Accuracy
    acc_fig, acc_ax = plt.subplots(figsize=(7, 5))
    acc_ax.plot(train_accs, label='Train Accuracy')
    acc_ax.plot(test_accs, label='Test Accuracy')
    acc_ax.set_title(f"{title} — Accuracy")
    acc_ax.set_xlabel('Epochs')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.legend()
    acc_ax.grid(True)

    if save_path:
        root, ext = os.path.splitext(save_path)
        loss_path = f"{root}_loss{ext}"
        acc_path = f"{root}_accuracy{ext}"
        os.makedirs(os.path.dirname(loss_path), exist_ok=True)
        loss_fig.savefig(loss_path)
        acc_fig.savefig(acc_path)

    # По умолчанию графики не выводятся на экран.
    if show:
        plt.show()
    plt.close(loss_fig)
    plt.close(acc_fig)

def plot_heatmap(data, title, save_path=None, show: bool = False):
    """Визуализирует данные в виде heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="viridis")
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()

def plot_weights_hist(weights, title, save_path=None, show=False):
    """Строит гистограмму распределения весов модели"""
    plt.figure(figsize=(8, 5))
    plt.hist(weights, bins=50, color='steelblue', alpha=0.9)
    plt.title(title)
    plt.xlabel('Значение веса')
    plt.ylabel('Частота')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()