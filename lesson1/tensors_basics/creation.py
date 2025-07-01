import torch
import numpy as np

# Создание тензора из списка
list_tensor = torch.tensor([[1, 2], [3, 4]])
print('Тензор из списка:', list_tensor)

# Создание тензора из numpy-массива
np_array = np.array([[5, 6], [7, 8]])
numpy_tensor = torch.from_numpy(np_array)
print('Тензор из numpy:', numpy_tensor)

# zeros, ones, arange, random
zeros_tensor = torch.zeros((2, 3))
print('zeros:', zeros_tensor)

ones_tensor = torch.ones((2, 3), dtype=torch.float32)
print('ones:', ones_tensor)

arange_tensor = torch.arange(0, 10, 2)
print('arange:', arange_tensor)

rand_tensor = torch.rand((2, 2))
print('rand:', rand_tensor)

# Задание типа и устройства
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
print('float64:', float_tensor)

if torch.cuda.is_available():
    cuda_tensor = torch.tensor([1, 2, 3], device='cuda')
    print('cuda tensor:', cuda_tensor) 