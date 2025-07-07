import torch
import logging
from tabulate import tabulate

from lesson4.homework.convolutional_basics.datasets import get_mnist_loaders, get_cifar_loaders

from models.cnn_models import (
    CNN_Kernel_3x3, CNN_Kernel_5x5, CNN_Kernel_7x7,
    VGGShallow, VGGMedium, VGGDeep, ModelWithBasicBlocks
)
from utils.training_utils import train_model
from utils.visualization_utils import plot_feature_maps
from utils.comparison_utils import plot_model_comparison


def count_parameters(model):
    """Подсчитывает количество обучемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#Логгер
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='results/architecture.log', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# Глобальная переменная для активации
activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


def run_kernel_analysis():
    logging.info("Анализ влияния размера ядра свертки")
    device = torch.device('cuda')
    train_loader, test_loader = get_mnist_loaders()

    models_to_train = {
        "CNN_Kernel_3x3": CNN_Kernel_3x3(),
        "CNN_Kernel_5x5": CNN_Kernel_5x5(),
        "CNN_Kernel_7x7": CNN_Kernel_7x7(),
    }

    histories = []
    results_data = []

    for name, model_instance in models_to_train.items():
        model = model_instance.to(device)
        logging.info(f"Обучение модели: {name}")
        params = count_parameters(model)
        logging.info(f"Параметры: {params:,}")

        # Визуализация активаций первого слоя
        hook = model.features[0].register_forward_hook(get_activation('conv1'))
        data, _ = next(iter(test_loader))
        _ = model(data.to(device))
        plot_feature_maps(activations['conv1'], f"{name}_conv1", save_dir='plots/architecture_analysis/kernel_size/')
        hook.remove()

        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        histories.append(history)
        final_acc = history['test_accs'][-1]
        results_data.append([name, f"{final_acc:.4%}", f"{params:,}"])

    headers = ["Модель", "Точность (Тест)", "Параметры"]
    logging.info("Результаты анализа размера ядра\n" + tabulate(results_data, headers=headers, tablefmt="grid"))
    plot_model_comparison(histories, list(models_to_train.keys()), metric='acc',
                          save_dir='plots/architecture_analysis/kernel_size/',
                          filename="сравнение_моделей_по_размеру_ядра.png")


def run_depth_analysis():
    """Анализ влияния глубины сети."""
    logging.info("Анализ влияния глубины сети ")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders()

    models_to_train = {
        "Shallow_2-layer": VGGShallow(),
        "Medium_4-layer": VGGMedium(),
        "Deep_6-layer": VGGDeep(),
        "Deep_With_Residuals": ModelWithBasicBlocks()
    }

    histories = []
    results_data = []

    for name, model_instance in models_to_train.items():
        model = model_instance.to(device)
        logging.info(f"Обучение модели: {name}")
        params = count_parameters(model)
        logging.info(f"Параметры: {params:,}")

        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        histories.append(history)
        final_acc = history['test_accs'][-1]
        results_data.append([name, f"{final_acc:.4%}", f"{params:,}"])

    # Визуализация карт признаков для самой глубокой модели c residual связями
    data, _ = next(iter(test_loader))
    deep_res_model = models_to_train["Deep_With_Residuals"].to(device)
    hook1 = deep_res_model.layer1[0].conv1.register_forward_hook(get_activation('layer1_conv1'))
    hook4 = deep_res_model.layer4[-1].conv2.register_forward_hook(get_activation('layer4_conv2'))
    _ = deep_res_model(data.to(device))
    plot_feature_maps(activations['layer1_conv1'], "Residual_layer1", save_dir='plots/architecture_analysis/depth/')
    plot_feature_maps(activations['layer4_conv2'], "Residual_layer4", save_dir='plots/architecture_analysis/depth/')
    hook1.remove()
    hook4.remove()
    plot_model_comparison(histories, list(models_to_train.keys()), metric='acc',
                          save_dir='plots/architecture_analysis/depth/', filename="сравнение_моделей_по_глубине.png")


if __name__ == '__main__':
    run_kernel_analysis()
    print()
    run_depth_analysis()