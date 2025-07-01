# Урок 1: Введение в PyTorch

В этом уроке мы познакомимся с основами работы с PyTorch, научимся создавать и использовать тензоры, а также рассмотрим базовые операции над ними.

## Цели урока
- Познакомиться с PyTorch и его возможностями
- Научиться создавать виртуальное окружение для Python-проектов
- Установить PyTorch
- Научиться работать с тензорами и выполнять базовые операции

---

## Гайд по созданию виртуального окружения и установке PyTorch

### 1. Создание виртуального окружения

Рекомендуется использовать `venv` (стандартный модуль Python):

```bash
python -m venv env
```

Для активации окружения:
- **Windows:**
  ```bash
  .\env\Scripts\activate
  ```
- **Linux/Mac:**
  ```bash
  source env/bin/activate
  ```

### 2. Установка PyTorch (актуальная версия 2.7.1)

#### CPU-версия:

```bash
pip install torch torchvision torchaudio
```

#### CUDA-версия (например, для CUDA 12.8):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

**Либо**

В `requirements.txt` добавьте:

```
--extra-index-url https://download.pytorch.org/whl/cu121
torch
torchvision
torchaudio
```

И установите с помощью:

```bash
pip install -r requirements.txt
```

Данная "фишка" позволяет вам формировать requirements.txt с учетом CUDA, что может быть полезно для ваших приложений.

---

## Дальнейшие шаги
- Изучите примеры работы с тензорами в папке `tensors_basics/` и других файлах урока. 