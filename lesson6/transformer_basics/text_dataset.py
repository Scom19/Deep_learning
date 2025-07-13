import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import os
from typing import List, Optional
import requests
import re


class TextDataset(Dataset):
    """Датасет для обучения генеративной модели на тексте"""

    def __init__(self, text_data: str, max_length: int = 192, tokenizer: Optional[Tokenizer] = None):
        self.max_length = max_length

        # Инициализируем токенизатор
        if tokenizer is None:
            tokenizer_path = os.path.join(os.path.dirname(__file__), "mistral_tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
        else:
            self.tokenizer = tokenizer

        # Специальные токены
        self.bos_token_id = self.tokenizer.token_to_id('<s>')
        self.eos_token_id = self.tokenizer.token_to_id('</s>')
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')

        # Подготавливаем текст
        self.sequences = self._prepare_sequences(text_data)

    def _prepare_sequences(self, text: str) -> List[torch.Tensor]:
        """Подготавливает последовательности для обучения"""
        sequences = []

        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 10:  # Пропускаем короткие абзацы
                continue

            # Токенизируем абзац
            tokens = self.tokenizer.encode(paragraph).ids

            # Добавляем BOS и EOS
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]

            # Создаём скользящие окна если текст длинный
            if len(tokens) > self.max_length:
                for i in range(0, len(tokens) - self.max_length + 1, self.max_length // 2):
                    window = tokens[i:i + self.max_length]
                    if len(window) == self.max_length:
                        sequences.append(torch.tensor(window, dtype=torch.long))
            else:
                sequences.append(torch.tensor(tokens, dtype=torch.long))

        return sequences

    def get_vocab_size(self):
        """Возвращает размер словаря"""
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self):
        """Возвращает ID токена padding"""
        return self.pad_token_id

    def get_bos_token_id(self):
        """Возвращает ID токена начала последовательности"""
        return self.bos_token_id

    def get_eos_token_id(self):
        """Возвращает ID токена конца последовательности"""
        return self.eos_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # Входная последовательность (все токены кроме последнего)
        input_ids = sequence[:-1]

        # Целевая последовательность (все токены кроме первого)
        target_ids = sequence[1:]

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


class TextCollator:
    """Коллатор для батчей TextDataset"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        target_ids = [item['target_ids'] for item in batch]

        # Делаем padding для всех последовательностей в батче
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_token_id)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


def download_text_from_url(url: str) -> str:
    """Загружает текст по URL с автоматическим определением кодировки"""
    response = requests.get(url)
    
    # Автоматически определяем кодировку
    try:
        import chardet
        detected = chardet.detect(response.content)
        if detected['encoding'] and detected['confidence'] > 0.7:
            response.encoding = detected['encoding']
        else:
            # Пробуем распространённые кодировки для русского текста
            encodings_to_try = ['utf-8', 'windows-1251', 'koi8-r', 'cp866']
            for encoding in encodings_to_try:
                try:
                    text = response.content.decode(encoding)
                    # Проверяем, что в тексте есть русские символы
                    if any(ord(c) >= 0x400 and ord(c) <= 0x4FF for c in text[:1000]):
                        response.encoding = encoding
                        break
                except UnicodeDecodeError:
                    continue
    except ImportError:
        # Если chardet не установлен, используем старый метод
        encodings_to_try = ['utf-8', 'windows-1251', 'koi8-r', 'cp866']
        for encoding in encodings_to_try:
            try:
                text = response.content.decode(encoding)
                # Проверяем, что в тексте есть русские символы
                if any(ord(c) >= 0x400 and ord(c) <= 0x4FF for c in text[:1000]):
                    response.encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
    
    response.raise_for_status()
    text = response.text
    return text.strip()


def download_sample_text() -> str:
    """Загружает русский текст для обучения"""
    # Александр Сергеевич Пушкин. Капитанская Дочка
    url = "http://lib.ru/LITRA/PUSHKIN/kapitan.txt"
    return download_text_from_url(url)


def clean_russian_text(text: str) -> str:
    """Очищает русский текст от служебной информации и HTML-тегов"""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    
    # Удаляем возможные заголовки и колонтитулы
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Пропускаем служебные строки
        if (line.startswith('http://') or
                line.startswith('www.') or
                line.startswith('lib.ru') or
                line.startswith('Текст взят') or
                line.startswith('Источник') or
                len(line) < 3):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Очищаем текст
    text = re.sub(r'\n{3,}', '\n\n', text)  # Максимум 2 переноса строки подряд
    text = re.sub(r'[ \t]+', ' ', text)  # Убираем лишние пробелы
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Убираем номера страниц

    return text


def create_text_dataloaders(
        text_data: str = None,
        text_url: str = None,
        batch_size: int = 1,
        max_length: int = 192,
        train_split: float = 0.9,
        tokenizer: Optional[Tokenizer] = None
):
    """Создаёт загрузчики данных для обучения генеративной модели
    
    Args:
        text_data: Готовый текст для обучения
        text_url: URL для загрузки текста (если text_data не задан)
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        train_split: Доля данных для обучения
        tokenizer: Токенизатор (если None, используется mistral_tokenizer.json)
    """

    if text_data is None:
        if text_url is not None:
            text_data = download_text_from_url(text_url)
        else:
            text_data = download_sample_text()

    # Очищаем текст от HTML-тегов и служебной информации
    text_data = clean_russian_text(text_data)

    # Создаём датасет
    dataset = TextDataset(text_data, max_length, tokenizer)

    # Разделяем на train и validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Создаём коллатор
    collator = TextCollator(dataset.get_pad_token_id())

    # Создаём загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, dataset
