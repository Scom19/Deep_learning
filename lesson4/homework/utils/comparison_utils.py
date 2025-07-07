import matplotlib.pyplot as plt
import os


def plot_model_comparison(histories, labels, metric='acc', save_dir='plots/', filename=None):
    """Сравнивает метрики нескольких моделей на одном графике."""
    metric_key = 'test_accs' if metric == 'acc' else 'test_losses'

    title_metric_rus = 'Точности' if metric == 'acc' else 'Потерь'
    y_label_rus = 'Точность' if metric == 'acc' else 'Потери'

    if filename is None:
        filename = f"сравнение_моделей_по_{title_metric_rus.lower()}.png"

    save_path = os.path.join(save_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 7))

    for history, label in zip(histories, labels):
        plt.plot(history[metric_key], label=label, marker='o', linestyle='--')

    plt.title(f'Сравнение моделей: {title_metric_rus}', fontsize=16)
    plt.xlabel('Эпоха')
    plt.ylabel(y_label_rus)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    print(f"График сравнения сохранен: {save_path}")
    plt.close()