import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

x_clone = x.clone()
print(f'x_clone = {x_clone}')

x_detach = x.detach()
print(f'x_detach = {x_detach}')

x_data = x.data
print(f'x_data = {x_data}')

x2 = torch.zeros_like(x)
x2.copy_(x)
print(f'x2 после copy_: {x2}') 