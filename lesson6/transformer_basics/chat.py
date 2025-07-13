from generator_transformer import GeneratorTransformer
import os

def chat():
    """
    Простой интерфейс для тестирования генеративного трансформера
    """
    checkpoint_path = "checkpoint.pt"
    
    # Проверяем наличие checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint файл '{checkpoint_path}' не найден!\n"
              f"Пожалуйста, сначала обучите модель используя train.py")
        return
    
    try:
        # Загружаем модель
        print("Загрузка модели...")
        device = 'cuda'
        model = GeneratorTransformer.load_from_checkpoint(checkpoint_path, device=device)
        model.eval()
        print(f"Модель успешно загружена на {device}")
        print("-" * 50)
        
        while True:
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() == 'quit':
                print("До свидания!")
                break
            
            if not user_input:
                print("Пожалуйста, введите текст")
                continue
                
            try:
                # Генерируем ответ
                print("Генерация...", end="", flush=True)
                response = model.generate(
                    user_input, 
                    max_out_tokens=50, 
                    temperature=0.8
                )
                print(f"\rБот: {response}")
                
            except Exception as e:
                print(f"\nОшибка при генерации: {e}")
                
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

if __name__ == "__main__":
    chat() 