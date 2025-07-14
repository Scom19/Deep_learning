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


def setup_logger(name='trainer'):
    """Настраивает логгер для обучения"""
    # Создаем папку для логов
    os.makedirs('./logs', exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики
    logger.handlers.clear()
    
    # Создаем файл для логов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'./logs/{name}_{timestamp}.log'
    
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
    """Выполняет одну эпоху обучения или тестирования модели"""
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
    
    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda', save_path='./weights/best_resnet18.pth', logger=None):
    """Обучает модель и сохраняет лучшие веса по точности"""
    if logger is None:
        logger = setup_logger('train_model')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0
    
    logger.info(f'Начинаем обучение модели. Количество эпох: {epochs}, Learning rate: {lr}')
    logger.info(f'Модель будет сохранена в: {save_path}')
    
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        logger.info('-' * 50)
        
        # Сохраняем лучшие веса
        if save_path is not None and test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f'Лучшие веса сохранены в {save_path} (Test Acc: {best_acc:.4f})')
    
    logger.info(f'Обучение завершено. Лучшая точность: {best_acc:.4f}')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def main():
    """Простой пример обучения модели ResNet-18"""
    logger = setup_logger('main')
    
    logger.info('Начинаем обучение модели ResNet-18')
    
    # Создаем датасеты с правильными путями
    train_dataset = CustomImageDataset(root_dir='./data/train', target_size=(224, 224))
    test_dataset = CustomImageDataset(root_dir='./data/test', target_size=(224, 224))
    
    logger.info(f'Загружены датасеты. Train: {len(train_dataset)} образцов, Test: {len(test_dataset)} образцов')
    
    # Уменьшаем batch_size для маленького датасета
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f'Размер батча: {batch_size}')
    
    # Создаем модель для 6 классов (героев)
    model = Resnet18(num_classes=6)
    logger.info('Создана модель ResNet-18 для 6 классов')
    
    # Обучаем модель
    train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cuda', logger=logger)


if __name__ == '__main__':
    main()