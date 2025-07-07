import torch
import logging
import time
from tabulate import tabulate

from lesson4.homework.convolutional_basics.datasets import get_cifar_loaders
from models.cnn_models import ModelWithBasicBlocks, ModelWithBottleneckBlocks

from models.custom_layers import Swish, AttentionMechanism, GatedConv2d, L2Pool2d

from utils.training_utils import train_model
from utils.comparison_utils import plot_model_comparison


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='results/custom_experiments.log', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)


def test_custom_layers():
    """Проверка работы кастомных слоев."""
    logging.info("Проверка кастомных слоев")
    dummy_tensor = torch.randn(4, 16, 32, 32)
    logging.info(f"Размер входного тензора: {dummy_tensor.shape}")

    # 1. Кастомная функция активации
    swish_layer = Swish()
    logging.info(f"Swish: {swish_layer(dummy_tensor).shape}")

    attention_layer = AttentionMechanism(in_channels=16)
    logging.info(f"Attention: {attention_layer(dummy_tensor).shape}")

    # 3. Кастомный сверточный слой
    gated_conv_layer = GatedConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    gated_conv_output = gated_conv_layer(dummy_tensor)
    logging.info(f"GatedConv2d: {gated_conv_output.shape}")
    # Проверяем, что количество каналов на выходе правильное
    assert gated_conv_output.shape[1] == 32

    # 4.pooling слой
    l2_pool_layer = L2Pool2d(kernel_size=2, stride=2)
    l2_pool_output = l2_pool_layer(dummy_tensor)
    logging.info(f"L2Pool2d: {l2_pool_output.shape}")
    # Проверяем, что пространственные размерности уменьшились в 2 раза
    assert l2_pool_output.shape[2] == 16 and l2_pool_output.shape[3] == 16
    logging.info("Кастомные слои работают корректно.")


def run_residual_block_experiments():
    logging.info("Сравнение различных Residual блоков")
    device = torch.device('cuda')

    train_loader, test_loader = get_cifar_loaders()

    models_to_train = {
        "ModelWithBasicBlocks": ModelWithBasicBlocks(),
        "ModelWithBottleneckBlocks": ModelWithBottleneckBlocks()
    }

    histories = []
    results_data = []

    for name, model_instance in models_to_train.items():
        model = model_instance.to(device)
        logging.info(f"Модель: {name}")
        params = count_parameters(model)
        logging.info(f"Количество параметров: {params:,}")
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device))
        elapsed_time = time.time() - start_time
        histories.append(history)
        final_acc = history['test_accs'][-1]
        results_data.append([name, f"{final_acc:.4%}", f"{params:,}", f"{elapsed_time:.2f}"])

    headers = ["Модель", "Точность", "Параметры", "Время обучения"]
    logging.info(
        "Результаты сравнения Residual блоков\n" + tabulate(results_data, headers=headers, tablefmt="grid"))

    plot_model_comparison(
        histories,
        list(models_to_train.keys()),
        metric='acc',
        save_dir='plots/custom_experiments/',
        filename="сравнение_типов_блоков_по_точности.png"
    )
    logging.info("Сравнение блоков завершено")


if __name__ == '__main__':
    test_custom_layers()
    print()
    run_residual_block_experiments()