import torch

x = torch.arange(16).reshape(4, 4)
print(f'x =\n{x}')

# Срезы
print(f'x[1:3, :] =\n{x[1:3, :]}')

# Индексация по списку
print(f'x[[0, 2], [1, 3]] = {x[[0, 2], [1, 3]]}')

# Логическая индексация
mask = x > 7
print(f'mask =\n{mask}')
print(f'x[x > 7] = {x[x > 7]}')

# Fancy indexing
idx = torch.tensor([0, 3])
print(f'x[idx] =\n{x[idx]}') 