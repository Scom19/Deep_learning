import matplotlib.pyplot as plt
import os
import seaborn as sns


def plot_training_history(history, model_name, save_dir='plots/'):
    """Сохраняет историю обучения модели"""
    save_path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training History for {model_name}', fontsize=16)

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['test_accs'], label='Test Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"График обучения сохранен: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name, save_dir='plots/'):
    """Сохраняет матрицу ошибок"""
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.tight_layout()
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(save_path)
    print(f"Матрица ошибок сохранена: {save_path}")
    plt.close()


def plot_feature_maps(feature_maps, layer_name, save_dir, max_maps=16):
    """Визуализирует и сохраняет карты признаков слоя"""
    save_path = os.path.join(save_dir, f"активации_{layer_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Берем активации для первого изображения в батче
    fm = feature_maps[0].detach().cpu()
    num_maps = min(fm.shape[0], max_maps)

    cols = 4
    rows = (num_maps + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3, rows * 3))
    fig.suptitle(f'Карты активаций слоя {layer_name}', fontsize=16)

    for i in range(num_maps):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(fm[i], cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Карта {i + 1}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    print(f"Карты активаций сохранены: {save_path}")
    plt.close()