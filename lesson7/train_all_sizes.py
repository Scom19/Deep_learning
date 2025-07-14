import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import logging
from datetime import datetime

from torchvision import transforms
from torch.utils.data import DataLoader

from core.datasets import CustomImageDataset
from core.model import Resnet18


def setup_logger(image_size):
    """Настраивает логгер для конкретного размера изображения"""
    # Создаем папку для логов
    os.makedirs('./logs', exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(f'training_{image_size}')
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики
    logger.handlers.clear()
    
    # Создаем файл для логов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'./logs/training_{image_size}_{timestamp}.log'
    
    # Создаем обработчик файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Создаем обработчик консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Создаем форматтер
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def run_epoch(model, data_loader, criterion, optimizer=None, device='cuda', is_test=False):
    """Выполняет одну эпоху обучения или тестирования"""
    if is_test:
        model.eval()
    else:
        model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        
        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total * 100
    
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda', save_path='./weights/best_resnet18.pth', logger=None):
    """Обучает модель и сохраняет лучшие веса"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_accuracy = 0
    
    model.to(device)
    
    logger.info(f'Начинаем обучение модели. Количество эпох: {epochs}, Learning rate: {lr}')
    logger.info(f'Модель будет сохранена в: {save_path}')
    
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch+1}/{epochs}')
        
        # Обучение
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        
        # Тестирование
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device=device, is_test=True)
        
        logger.info(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            logger.info(f'Новая лучшая модель сохранена с точностью {best_accuracy:.2f}%')
        
        logger.info('---')
    
    logger.info(f'Обучение завершено. Итоговая лучшая точность: {best_accuracy:.2f}%')
    
    return model, best_accuracy


def main():
    """Обучает модели ResNet-18 для разных размеров изображений"""
    image_sizes = [224, 256, 384, 512]
    epochs = 10
    lr = 0.001
    device = 'cuda'
    
    # Создаем папку для весов
    os.makedirs('./weights', exist_ok=True)
    
    # Создаем общий логгер
    main_logger = setup_logger('main')
    
    main_logger.info('Начинаем обучение моделей для разных размеров изображений')
    main_logger.info(f'Размеры изображений: {image_sizes}')
    main_logger.info(f'Количество эпох: {epochs}')
    main_logger.info(f'Learning rate: {lr}')
    main_logger.info(f'Устройство: {device}')
    
    for size in image_sizes:
        main_logger.info(f"Обучение модели для размера изображения {size}x{size}")
        
        # Создаем отдельный логгер для каждого размера
        size_logger = setup_logger(size)
        
        # Создаем модель для 6 классов (героев)
        model = Resnet18(num_classes=6)
        size_logger.info(f'Создана модель ResNet-18 для {size}x{size} изображений')
        
        # Создаем датасеты с правильными путями
        train_dataset = CustomImageDataset(root_dir='./data/train', target_size=(size, size))
        test_dataset = CustomImageDataset(root_dir='./data/test', target_size=(size, size))
        
        size_logger.info(f'Загружены датасеты. Train: {len(train_dataset)} образцов, Test: {len(test_dataset)} образцов')
        
        # Уменьшаем batch_size для маленького датасета
        batch_size = 16
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        size_logger.info(f'Размер батча: {batch_size}')
        
        # Обучаем модель
        model, best_accuracy = train_model(
            model, train_loader, test_loader, 
            epochs=epochs, lr=lr, device=device,
            save_path=f'./weights/best_resnet18_{size}.pth',
            logger=size_logger
        )
        
        main_logger.info(f"Обучение для размера {size}x{size} завершено. Лучшая точность: {best_accuracy:.2f}%")
    
    main_logger.info('Все модели обучены!')


if __name__ == '__main__':
    main() 