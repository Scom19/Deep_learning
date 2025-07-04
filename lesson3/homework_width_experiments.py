import sys
import os
import pandas as pd
import logging

sys.path.append(os.path.dirname(__file__))

from utils.experiment_utils import run_experiment
from utils.visualization_utils import plot_heatmap
from fully_connected_basics.datasets import get_mnist_loaders

logger = logging.getLogger(__name__)

def width_experiments(train_loader, test_loader, *, epochs: int = 10):
    """Влияние ширины скрытых слоёв на модель"""
    presets = {
        "narrow": [64, 32, 16],
        "medium": [256, 128, 64],
        "wide": [1024, 512, 256],
        "very_wide": [2048, 1024, 512],
    }

    plans = [
        {
            "name": f"width_{tag}",
            "experiment_type": "width_experiments",
            "input_size": 784,
            "num_classes": 10,
            "layers": [
                {"type": "linear", "size": cfg[0]}, {"type": "relu"},
                {"type": "linear", "size": cfg[1]}, {"type": "relu"},
                {"type": "linear", "size": cfg[2]}, {"type": "relu"},
            ],
        }
        for tag, cfg in presets.items()
    ]

    summary = [
        run_experiment(p, train_loader, test_loader, epochs=epochs) for p in plans
    ]

    for res in summary:
        logger.info(
            f"{res['config']['name']:<20} | "
            f"Acc={res['final_test_accuracy']:.4f} "
            f"| Time={res['training_time']:.1f}s "
            f"| Params={res['num_parameters']}"
        )
    # Grid search
    schemas = {
        "expanding": [128, 256, 512],
        "contracting": [512, 256, 128],
        "constant": [256, 256, 256],
    }

    grid_outcomes = [
        run_experiment(
            {
                "name": f"grid_{label}",
                "experiment_type": "width_experiments",
                "input_size": 784,
                "num_classes": 10,
                "layers": [
                    {"type": "linear", "size": spec[0]}, {"type": "relu"},
                    {"type": "linear", "size": spec[1]}, {"type": "relu"},
                    {"type": "linear", "size": spec[2]}, {"type": "relu"},
                ],
            },
            train_loader,
            test_loader,
            epochs=5,
        )
        for label, spec in schemas.items()
    ]

    # Heatmap визуализаация
    df = pd.DataFrame({
        "Scheme": [r['config']['name'] for r in grid_outcomes],
        "Accuracy": [r['final_test_accuracy'] for r in grid_outcomes],
    }).set_index("Scheme")

    plot_heatmap(df, "Grid-Search: Accuracy", "plots/width_experiments/grid_search_heatmap.png")


if __name__ == '__main__':
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    width_experiments(train_loader, test_loader, epochs=10)