import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(f'x до inplace: {x}')

x.add_(5)
print(f'x после add_: {x}')

x.mul_(2)
print(f'x после mul_: {x}')

x.zero_()
print(f'x после zero_: {x}')

# Обычные операции не меняют исходный тензор
x = torch.tensor([1.0, 2.0, 3.0])
y = x + 10
print(f'x после x + 10: {x}, y = {y}') 