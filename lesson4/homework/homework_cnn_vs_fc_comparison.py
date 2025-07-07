import torch
import time
import logging
from tabulate import tabulate
from sklearn.metrics import confusion_matrix

from lesson4.homework.convolutional_basics.datasets import get_mnist_loaders, get_cifar_loaders
from models.fc_models import FC_MNIST, FC_CIFAR

from models.cnn_models import SimpleCNN, ModelWithBasicBlocks, ModelWithBasicBlocksAndDropout
from utils.training_utils import train_model, get_predictions
from utils.visualization_utils import plot_training_history, plot_confusion_matrix
from utils.comparison_utils import plot_model_comparison

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Логгер
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='results/comparison.log', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

def run_mnist_comparison():
    logging.info("Сравнение моделей на MNIST")
    device = torch.device('cuda')
    train_loader, test_loader = get_mnist_loaders()

    models_to_train = {
        "FC_MNIST": FC_MNIST(),
        "SimpleCNN": SimpleCNN(),
        # Модель для 1-канальных изображений
        "ModelWithBasicBlocks": ModelWithBasicBlocks(input_channels=1)
    }

    histories = []; results_data = []
    for name, model_instance in models_to_train.items():
        model = model_instance.to(device)
        logging.info(f"Обучение модели: {name}")
        params = count_parameters(model); logging.info(f"Параметры: {params:,}")
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        elapsed_time = time.time() - start_time
        histories.append(history)
        final_acc = history['test_accs'][-1]
        results_data.append([name, f"{final_acc:.4%}", f"{params:,}", f"{elapsed_time:.2f}"])
        plot_training_history(history, model_name=name, save_dir='plots/mnist_comparison/')
    headers = ["Модель (MNIST)", "Точность", "Параметры", "Время обучения (сек)"]; logging.info("Результаты сравнения на MNIST\n" + tabulate(results_data, headers=headers, tablefmt="grid"))
    plot_model_comparison(histories, list(models_to_train.keys()), metric='acc', save_dir='plots/mnist_comparison/', filename="сравнение_моделей_MNIST_по_точности.png")
    plot_model_comparison(histories, list(models_to_train.keys()), metric='loss', save_dir='plots/mnist_comparison/', filename="сравнение_моделей_MNIST_по_потерям.png")


def run_cifar_comparison():
    logging.info("Сравнение моделей на CIFAR-10")
    device = torch.device('cuda')
    train_loader, test_loader = get_cifar_loaders()
    cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    models_to_train = {
        "FC_CIFAR": FC_CIFAR(),
        "ModelWithBasicBlocks": ModelWithBasicBlocks(),
        "ModelWithRegularization": ModelWithBasicBlocksAndDropout()
    }

    histories = []; results_data = []; best_acc = 0.0; best_model = None; best_model_name = ""
    for name, model_instance in models_to_train.items():
        model = model_instance.to(device)
        logging.info(f"Обучение модели: {name}")
        params = count_parameters(model); logging.info(f"Количество параметров: {params:,}")
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        elapsed_time = time.time() - start_time
        histories.append(history)
        final_acc = history['test_accs'][-1]; train_test_gap = history['train_accs'][-1] - final_acc
        results_data.append([name, f"{final_acc:.4%}", f"{train_test_gap:.4%}", f"{params:,}", f"{elapsed_time:.2f}"])
        plot_training_history(history, model_name=name, save_dir='plots/cifar_comparison/')
        if final_acc > best_acc: best_acc = final_acc; best_model = model; best_model_name = name
    headers = ["Модель (CIFAR-10)", "Точность", "Разрыв (Train-Test)", "Параметры", "Время обучения (сек)"]; logging.info("Результаты сравнения на CIFAR-10\n" + tabulate(results_data, headers=headers, tablefmt="grid"))
    plot_model_comparison(histories, list(models_to_train.keys()), metric='acc', save_dir='plots/cifar_comparison/', filename="сравнение_моделей_CIFAR-10_по_точности.png")
    plot_model_comparison(histories, list(models_to_train.keys()), metric='loss', save_dir='plots/cifar_comparison/', filename="сравнение_моделей_CIFAR-10_по_потерям.png")
    logging.info(f"Построение матрицы ошибок для лучшей модели: {best_model_name}")
    preds, targets = get_predictions(best_model, test_loader, device); cm = confusion_matrix(targets, preds)
    plot_confusion_matrix(cm, class_names=cifar_classes, model_name=best_model_name, save_dir='plots/cifar_comparison/')
    logging.info("Матрица ошибок сохранена.")


if __name__ == '__main__':
    run_mnist_comparison()
    print()
    run_cifar_comparison()