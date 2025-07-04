import logging

from lesson3.fully_connected_basics.datasets import get_mnist_loaders
from lesson3.utils.experiment_utils import run_experiment

logger = logging.getLogger(__name__)

def _make_dense_block(units: int, *, activation: bool = True):
    #Создаёт пару Linear-ReLU для уменьшения дублирования кода.
    block = [{"type": "linear", "size": units}]
    if activation:
        block.append({"type": "relu"})
    return block


def depth_experiments(train_loader, test_loader, *, epochs: int = 10) -> None:
    #Запускает серию экспериментов для оценки влияния глубины сети на качество модели
    hidden = 128  # базовый размер скрытого слоя
    # уникальное_имя, количество_скрытых_слоёв
    blueprint = [
        ("depth_1_layer", 0),
        ("depth_2_layers", 1),
        ("depth_3_layers", 2),
        ("depth_5_layers", 4),
        ("depth_7_layers", 6),
    ]

    experiment_plans = []
    for name, n_hidden in blueprint:
        layers = []
        for _ in range(n_hidden):
            layers.extend(_make_dense_block(hidden))
        experiment_plans.append(
            {
                "name": name,
                "experiment_type": "depth_experiments",
                "input_size": 784,
                "num_classes": 10,
                "layers": layers,
            }
        )

    #эксперимент с регуляризацией
    experiment_plans.append(
        {
            "name": "depth_5_layers_regularized",
            "experiment_type": "depth_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": (
                (
                    _make_dense_block(hidden, activation=False)
                    + [{"type": "batch_norm"}, {"type": "relu"}, {"type": "dropout", "rate": 0.3}]
                )
                * 3  # повторяем блок трижды
                + _make_dense_block(hidden)  # завершающий блок
            ),
        }
    )

    all_results = [
        run_experiment(cfg, train_loader, test_loader, epochs=epochs)
        for cfg in experiment_plans
    ]

    # Выводим сводку
    for res in all_results:
        logger.info(
            f"{res['config']['name']:<30} | "
            f"Acc={res['final_test_accuracy']:.4f} "
            f"| Time={res['training_time']:.1f}s "
            f"| Params={res['num_parameters']}"
        )

    #Определение оптимальной глубины
    best_run = max(
        (r for r in all_results if not r['config']['name'].endswith('_regularized')),
        key=lambda r: r['final_test_accuracy'],
    )
    logger.info(
        "\nЛучший результат по глубине: "
        f"{best_run['config']['name']} с accuracy {best_run['final_test_accuracy']:.4f}"
    )


if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    depth_experiments(train_loader, test_loader, epochs=10)