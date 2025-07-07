import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Кастомная функция активации Swish"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class AttentionMechanism(nn.Module):
    """Простой Channel Attention"""
    def __init__(self, in_channels, reduction=4):
        super(AttentionMechanism, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c); y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConv2d, self).__init__()
        # Свертка производит в 2 раза больше каналов, чем нужно на выходе
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, stride, padding)

    def forward(self, x):
        #Применяем свертку
        x_conv = self.conv(x)
        #елим тензор на две части по оси каналов
        out, gate = x_conv.chunk(2, dim=1)
        # Одна часть становится данными, вторая - гейтом
        return out * torch.sigmoid(gate)

class L2Pool2d(nn.Module):
    """
    Кастомный pooling слой
    Он вычисляет корень из среднего квадратов значений в окне
    """
    def __init__(self, kernel_size, stride=None):
        super(L2Pool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        # Вычисляем среднее значение квадратов с помощью стандартного AvgPool2d
        squared_avg = F.avg_pool2d(x.pow(2), kernel_size=self.kernel_size, stride=self.stride)
        # Добавляем эпсилон для численной стабильности (чтобы избежать sqrt(0))
        return torch.sqrt(squared_avg + 1e-6)

class BottleneckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)); out += self.shortcut(x)
        return F.relu(out)