import sys
import os
import logging

sys.path.append(os.path.dirname(__file__))

from utils.experiment_utils import run_experiment
from fully_connected_basics.datasets import get_mnist_loaders

logger = logging.getLogger(__name__)


def _attach_dropout(rate: float):
    return [{"type": "dropout", "rate": rate}]


def regularization_experiments(train_loader, test_loader, *, epochs: int = 15):
    #Сравнивает различные способы регуляризации

    def base_layers():
        """Базовая архитектура"""
        return [
            {"type": "linear", "size": 512}, {"type": "relu"},
            {"type": "linear", "size": 256}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "relu"},
        ]

    plans = [
        {
            "name": "reg_none",
            "experiment_type": "regularization_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": base_layers(),
            "weight_decay": 0.0,
        },

        {
            "name": "reg_batchnorm",
            "experiment_type": "regularization_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": 512}, {"type": "batch_norm"}, {"type": "relu"},
                {"type": "linear", "size": 256}, {"type": "batch_norm"}, {"type": "relu"},
                {"type": "linear", "size": 128}, {"type": "batch_norm"}, {"type": "relu"},
            ],
            "weight_decay": 0.0,
        },
        {
            "name": "reg_dropout_batchnorm",
            "experiment_type": "regularization_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": 512}, {"type": "batch_norm"}, {"type": "relu"}, *_attach_dropout(0.3),
                {"type": "linear", "size": 256}, {"type": "batch_norm"}, {"type": "relu"}, *_attach_dropout(0.3),
                {"type": "linear", "size": 128}, {"type": "batch_norm"}, {"type": "relu"},
            ],
            "weight_decay": 0.0,
        },
        {
            "name": "reg_l2",
            "experiment_type": "regularization_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": base_layers(),
            "weight_decay": 1e-4,
        },
    ]

    for rate in (0.1, 0.3, 0.5):
        plans.append(
            {
                "name": f"reg_dropout_{rate}",
                "experiment_type": "regularization_experiments",
                "input_size": 784,
                "num_classes": 10,
                "layers": [
                    {"type": "linear", "size": 512}, {"type": "relu"}, *_attach_dropout(rate),
                    {"type": "linear", "size": 256}, {"type": "relu"}, *_attach_dropout(rate),
                    {"type": "linear", "size": 128}, {"type": "relu"},
                ],
                "weight_decay": 0.0,
            }
        )

    # Переменный dropout: 0.5 → 0.3 → 0.1
    plans.append(
        {
            "name": "reg_dropout_variable",
            "experiment_type": "regularization_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": 512}, {"type": "relu"}, *_attach_dropout(0.5),
                {"type": "linear", "size": 256}, {"type": "relu"}, *_attach_dropout(0.3),
                {"type": "linear", "size": 128}, {"type": "relu"}, *_attach_dropout(0.1),
            ],
            "weight_decay": 0.0,
        }
    )

    results = [
        run_experiment(p, train_loader, test_loader, epochs=epochs, weight_decay=p.get("weight_decay", 0.0))
        for p in plans
    ]

    logger.info("\nСводка по регуляризации:")
    for res in results:
        logger.info(f"{res['config']['name']:<25} | Acc={res['final_test_accuracy']:.4f} | Time={res['training_time']:.1f}s")

if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    regularization_experiments(train_loader, test_loader, epochs=15)