import torch

x = torch.randn(2, 3, 4)
print(f'x.shape = {x.shape}')
print(f'x.size() = {x.size()}')
print(f'x.numel() = {x.numel()}')
print(f'x.dim() = {x.dim()}')
print(f'x.ndim = {x.ndim}')

scalar = torch.tensor(42.0)
print(f'scalar.item() = {scalar.item()}') 