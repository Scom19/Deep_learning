import torch
import torch.nn as nn
from tokenizers import Tokenizer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from generator_transformer import GeneratorTransformer
from text_dataset import download_sample_text, create_text_dataloaders, clean_russian_text

def setup_logging():
    """Настройка логирования в файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # Настройка логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Для вывода в консоль тоже
        ]
    )
    
    return log_filename

def train_model():
    """Обучение генеративной модели трансформера"""
    
    # Настройка логирования
    log_filename = setup_logging()
    logger = logging.getLogger(__name__)
    
    # Параметры обучения
    batch_size = 1
    max_length = 192
    learning_rate = 1e-4
    num_epochs = 10
    device = 'cuda'
    
    logger.info(f"Параметры обучения:")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Max length: {max_length}")
    logger.info(f"- Learning rate: {learning_rate}")
    logger.info(f"- Epochs: {num_epochs}")
    logger.info(f"- Device: {device}")
    
    # Загружаем токенизатор
    tokenizer_path = "mistral_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Токенизатор {tokenizer_path} не найден!")
        return
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
    logger.info(f"Токенизатор загружен из {tokenizer_path}")
    
    # Загружаем или создаём данные
    logger.info("Загрузка русского текста...")
    text_data = download_sample_text()
    logger.info(f"Загружено {len(text_data)} символов русского текста")
    
    # Очищаем текст от HTML-тегов и служебной информации
    text_data = clean_russian_text(text_data)
    logger.info(f"После очистки: {len(text_data)} символов")
    logger.info(f"Первые 200 символов (после очистки): {text_data[:200]}...")
    
    # Создаём датасет и даталоадер
    train_loader, val_loader, dataset = create_text_dataloaders(
        text_data=text_data,
        batch_size=batch_size,
        max_length=max_length,
        tokenizer=tokenizer
    )
    
    logger.info(f"Создано {len(train_loader)} батчей для обучения")
    logger.info(f"Создано {len(val_loader)} батчей для валидации")
    
    # Создаём модель
    model = GeneratorTransformer(
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=2,
        vocab_size=dataset.get_vocab_size(),
        max_length=max_length,
        tokenizer=tokenizer,
        device=device
    )
    
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Модель создана с {num_params:,} параметрами")
    
    # Оптимизатор и функция потерь
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))
    
    # Списки для сохранения метрик
    train_losses = []
    val_losses = []
    
    # Цикл обучения
    
    for epoch in range(num_epochs):
        
        model.train()
        total_train_loss = 0
        
        # Обучение
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")
        for batch_idx, batch in enumerate(train_pbar):
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            logits = model(inputs, targets)
            
            # Вычисляем потери
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val")
            for batch in val_pbar:
                inputs = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
                
                logits = model(inputs, targets)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        logger.info(f"Результаты эпохи {epoch+1}:")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        
        # Тестируем генерацию
        model.eval()
        test_prompt = "Эх,  батюшка"
        generated = model.generate(test_prompt, max_out_tokens=80, temperature=0.8)
        logger.info(f"Тестовая Промт'{test_prompt}'. Ответ:{generated}")
    
    # Сохраняем модель
    checkpoint_path = "checkpoint.pt"
    model.save_checkpoint(checkpoint_path)
    logger.info(f"\nМодель сохранена в {checkpoint_path}")
    
    # Сохраняем график потерь
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    logger.info("График потерь сохранен в training_loss.png")

    
    model.eval()
    test_prompts = [
        "Отец мой",
        "Помилуй, батюшка",
        "он  рассказывал  мне"
    ]
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_out_tokens=50, temperature=0.8)
        logger.info(f"Промпт: '{prompt}'")
        logger.info(f"Генерация: {generated}\n")
    
    # Итоговая статистика
    logger.info(f"Финальная train loss: {train_losses[-1]:.4f}")
    logger.info(f"Финальная val loss: {val_losses[-1]:.4f}")
    logger.info(f"Минимальная train loss: {min(train_losses):.4f} (эпоха {train_losses.index(min(train_losses))+1})")
    logger.info(f"Минимальная val loss: {min(val_losses):.4f} (эпоха {val_losses.index(min(val_losses))+1})")
    logger.info(f"Всего параметров модели: {num_params:,}")
    logger.info(f"Размер словаря: {dataset.get_vocab_size()}")

    logger.info(f"Лог сохранен в файл: {log_filename}")

if __name__ == "__main__":
    train_model() 