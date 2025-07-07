import torch
import torch.nn as nn
import torch.nn.functional as F
from lesson4.homework.models.custom_layers import BottleneckResidualBlock

class ResidualBlock(nn.Module):
    """
    Базовый остаточный блок с двумя сверточными слоями
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ModelWithBasicBlocks(nn.Module):
    """Модель с базовыми остаточными блоками."""
    def __init__(self, num_blocks=[2,2,2,2], num_classes=10, input_channels=3):
        super(ModelWithBasicBlocks, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * ResidualBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)

class ModelWithBasicBlocksAndDropout(ModelWithBasicBlocks):
    """Модель с базовыми остаточными блоками и Dropout."""
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        return self.linear(out)

class ModelWithBottleneckBlocks(nn.Module):
    """Модель, использующая BottleneckResidualBlock"""
    def __init__(self, num_blocks=[2,2,2,2], num_classes=10, input_channels=3):
        super(ModelWithBottleneckBlocks, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BottleneckResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BottleneckResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BottleneckResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BottleneckResidualBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * BottleneckResidualBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)

class SimpleCNN(nn.Module):
    """Небольшая сверточная сеть для классификации изображений."""
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class CNN_Kernel_3x3(nn.Module):
    """CNN с ядрами 3x3."""
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN_Kernel_3x3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2) )
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1); return self.fc(x)

class CNN_Kernel_5x5(nn.Module):
    """CNN с ядрами 5x5."""
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN_Kernel_5x5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 20, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2) )
        self.fc = nn.Linear(40 * 7 * 7, num_classes)
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1); return self.fc(x)

class CNN_Kernel_7x7(nn.Module):
    """CNN с ядрами 7x7."""
    def __init__(self, input_channels=1, num_classes=10):
        super(CNN_Kernel_7x7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 13, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(13, 26, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool2d(2, 2) )
        self.fc = nn.Linear(26 * 7 * 7, num_classes)
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1); return self.fc(x)

# Конфигурации и классы для анализа глубины
cfg = {
    'Shallow': [64, 'M', 128, 'M'],
    'Medium':  [64, 64, 'M', 128, 128, 'M'],
    'Deep':    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
}

def make_layers(cfg, batch_norm=True):
    layers = []; in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] if batch_norm else [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_style_CNN(nn.Module):
    """Универсальная модель для анализа глубины."""
    def __init__(self, features, num_classes=10, final_channels=256):
        super(VGG_style_CNN, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * 4 * 4, 512),
            nn.ReLU(True), nn.Dropout(), nn.Linear(512, num_classes), )
    def forward(self, x):
        x = self.features(x); x = x.view(x.size(0), -1); return self.classifier(x)


class VGGShallow(nn.Module):
    """Неглубокая модель (2 сверточных слоя)."""

    def __init__(self, num_classes=10):
        super(VGGShallow, self).__init__()
        # Конфигурация: 2 свертки, 2 пулинга -> карта признаков 8x8
        self.features = make_layers([64, 'M', 128, 'M'])
        # Размер входа классификатора: 128 каналов * 8 * 8
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGGMedium(nn.Module):
    """Средняя модель (4 сверточных слоя)."""

    def __init__(self, num_classes=10):
        super(VGGMedium, self).__init__()
        # Конфигурация: 4 свертки, 2 пулинга -> карта признаков 8x8
        self.features = make_layers([64, 64, 'M', 128, 128, 'M'])
        # Размер входа классификатора: 128 каналов * 8 * 8
        self.classifier = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VGGDeep(nn.Module):
    """Глубокая модель (6 сверточных слоев)."""

    def __init__(self, num_classes=10):
        super(VGGDeep, self).__init__()
        # Конфигурация: 6 сверток, 3 пулинга -> карта признаков 4x4
        self.features = make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M'])
        # Размер входа классификатора: 256 каналов * 4 * 4
        self.classifier = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)