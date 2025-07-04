import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = get_mnist_loaders(batch_size=64)

model = FullyConnectedModel(
    input_size=784,
    num_classes=10,
    layers=[
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "linear", "size": 128},
        {"type": "relu"}
    ]
).to(device)

print(f"Model parameters: {count_parameters(model)}")

history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))

plot_training_history(history) 