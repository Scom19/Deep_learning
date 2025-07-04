import torch
import torch.nn as nn

class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, num_classes, layer_config):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = self._build_layers(layer_config)

    def _build_layers(self, layer_config):
        layers = []
        prev_size = self.input_size

        for layer_spec in layer_config:
            layer_type = layer_spec['type']

            if layer_type == 'linear':
                out_size = layer_spec['size']
                layers.append(nn.Linear(prev_size, out_size))
                prev_size = out_size
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'tanh':
                layers.append(nn.Tanh())
            elif layer_type == 'dropout':
                rate = layer_spec.get('rate', 0.5)
                layers.append(nn.Dropout(rate))
            elif layer_type == 'batch_norm':
                layers.append(nn.BatchNorm1d(prev_size))
            elif layer_type == 'layer_norm':
                layers.append(nn.LayerNorm(prev_size))

        layers.append(nn.Linear(prev_size, self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)