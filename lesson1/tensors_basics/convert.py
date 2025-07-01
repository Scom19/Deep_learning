import torch
import numpy as np

# Из списка в тензор
lst = [1, 2, 3]
t = torch.tensor(lst)
print(f'из списка: {t}, dtype={t.dtype}')

# Из numpy в тензор
arr = np.array([1, 2, 3], dtype=np.float64)
t2 = torch.from_numpy(arr)
print(f'из numpy float64: {t2}, dtype={t2.dtype}')

# float64 → float32
f32 = t2.float()
print(f'float64 → float32: {f32}, dtype={f32.dtype}')

# float32 → float64
f64 = f32.double()
print(f'float32 → float64: {f64}, dtype={f64.dtype}')

# Из torch в numpy
arr2 = t2.numpy()
print(f'в numpy: {arr2}, dtype={arr2.dtype}')

# Из torch в список
lst2 = t2.tolist()
print(f'в список: {lst2}, type={type(lst2)}') 