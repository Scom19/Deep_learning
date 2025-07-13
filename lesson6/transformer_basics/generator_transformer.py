import torch
import torch.nn as nn
from tokenizers import Tokenizer
from typing import Optional
import os

from layers import DecoderLayer, MultiheadAttention, FeedForward, Embedding

class DecoderOnlyLayer(nn.Module):
    """Упрощённый слой декодера"""
    
    def __init__(self, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attention = mha
        self.ffn = ffn
        self.layernorm1 = nn.LayerNorm(mha.d_model)
        self.layernorm2 = nn.LayerNorm(mha.d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention с residual connection
        x_norm = self.layernorm1(x)
        x = x + self.self_attention(x_norm, x_norm, x_norm, mask)[0]
        
        # Feed-forward с residual connection
        x_norm = self.layernorm2(x)
        x = self.dropout(x + self.ffn(x_norm))
        
        return x

def get_subsequent_mask(x: torch.Tensor):
    """Создаёт маску для предотвращения доступа к будущим токенам"""
    batch_size, seq_len = x.size()
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

def get_pad_mask(x: torch.Tensor, pad_index: int):
    """Создаёт маску для игнорирования pad токенов"""
    return (x != pad_index).unsqueeze(-2)

class GeneratorTransformer(nn.Module):
    """трансформер для генерации текста"""
    
    def __init__(
        self, 
        d_model: int = 256, 
        num_heads: int = 8, 
        d_ff: int = 512, 
        num_layers: int = 6, 
        vocab_size: int = 32000, 
        pad_index: int = 1, 
        dropout: float = 0.1,
        max_length: int = 192,
        tokenizer: Optional[Tokenizer] = None,
        device: str = 'cuda',
    ):
        super().__init__()

        # Создаём упрощённые слои декодера
        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(
                mha=MultiheadAttention(d_model, num_heads, dropout),
                ffn=FeedForward(d_model, d_ff, dropout),
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Эмбеддинги и проекция в словарь
        self.embedding = Embedding(d_model, vocab_size, pad_index)
        self.normalize = nn.LayerNorm(d_model)
        self.vocab_projection = nn.Linear(d_model, vocab_size)
        
        # Настройки модели
        self.d_model = d_model
        self.max_length = max_length
        self.pad_index = pad_index
        self.device = device
        
        # Токенизатор
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
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Прямой проход через декодер"""
        # Создаём маски
        pad_mask = get_pad_mask(x, self.pad_index)
        subsequent_mask = get_subsequent_mask(x)
        combined_mask = pad_mask & subsequent_mask.to(self.device)
        
        # Получаем эмбеддинги
        x = self.embedding(x)
        
        # Проходим через слои декодера
        for layer in self.decoder_layers:
            x = layer.forward(x, combined_mask)
            
        # Нормализация и проекция в словарь
        x = self.normalize(x)
        logits = self.vocab_projection(x)
        
        return logits
    
    def generate(self, prompt: str, max_out_tokens: int = 200, temperature: float = 1.0):
        """Авторегрессивная генерация текста"""
        self.eval()
        
        with torch.no_grad():
            # Токенизируем промпт
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            generated = input_ids.clone()
            
            for _ in range(max_out_tokens):
                # Ограничиваем контекст максимальной длиной
                current_input = generated[:, -self.max_length:]
                
                # Получаем предсказание
                logits = self.forward(current_input)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Выбираем следующий токен
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 1
                )
                
                # Добавляем к результату
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Проверяем на EOS
                if next_token.item() == self.eos_token_id:
                    break
                    
            return self.tokenizer.decode(generated[0].tolist())
    
    def save_checkpoint(self, path: str):
        """Сохранение модели"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'max_length': self.max_length,
                'pad_index': self.pad_index,
                'num_layers': len(self.decoder_layers),
                'num_heads': self.decoder_layers[0].self_attention.num_heads,
                'd_ff': self.decoder_layers[0].ffn.d_ff,
                'vocab_size': self.vocab_projection.out_features,
            }
        }
        torch.save(checkpoint, path)
        print(f"Модель сохранена в {path}")
    
    @classmethod
    def load_from_checkpoint(cls, path: str, device: str = 'cuda'):
        """Загрузка модели из checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        # Создаём модель с конфигурацией из checkpoint
        config = checkpoint['model_config']
        model = cls(
            d_model=config['d_model'],
            max_length=config['max_length'],
            pad_index=config['pad_index'],
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            d_ff=config.get('d_ff', 512),
            vocab_size=config.get('vocab_size', 32000),
            device=device
        )
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model 