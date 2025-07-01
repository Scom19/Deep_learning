import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f'x.requires_grad = {x.requires_grad}')

with torch.no_grad():
    y = x * 2
    print(f'y (no grad) = {y}, y.requires_grad = {y.requires_grad}')

x2 = x.detach().requires_grad_(False)
print(f'x2.requires_grad = {x2.requires_grad}') 