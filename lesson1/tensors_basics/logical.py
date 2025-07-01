import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 2, 2])

print(f'x == y: {x == y}')
print(f'x > y: {x > y}')
print(f'x < y: {x < y}')
print(f'torch.eq(x, y): {torch.eq(x, y)}')
print(f'torch.gt(x, y): {torch.gt(x, y)}')
print(f'torch.lt(x, y): {torch.lt(x, y)}')

print(f'torch.any(x > 1): {torch.any(x > 1)}')
print(f'torch.all(x > 0): {torch.all(x > 0)}') 