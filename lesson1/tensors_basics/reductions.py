import torch

x = torch.arange(1, 7).reshape(2, 3).float()
print(f'x =\n{x}')

print(f'sum: {x.sum()}')
print(f'mean: {x.mean()}')
print(f'min: {x.min()}')
print(f'max: {x.max()}')
print(f'argmax: {x.argmax()}')
print(f'argmin: {x.argmin()}')
print(f'prod: {x.prod()}')
print(f'std: {x.std()}')
print(f'var: {x.var()}')
print(f'cumsum: {x.cumsum(dim=1)}')
print(f'cumprod: {x.cumprod(dim=1)}') 