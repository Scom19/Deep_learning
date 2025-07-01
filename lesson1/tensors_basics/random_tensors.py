import torch

# Фиксируем seed для воспроизводимости
torch.manual_seed(42)

x = torch.rand(2, 3)
print(f'rand: {x}')

x_like = torch.rand_like(x)
print(f'rand_like: {x_like}')

x_int = torch.randint(0, 10, (2, 3))
print(f'randint: {x_int}')

x_norm = torch.randn(2, 3)
print(f'randn: {x_norm}')

x_perm = torch.randperm(10)
print(f'randperm: {x_perm}') 