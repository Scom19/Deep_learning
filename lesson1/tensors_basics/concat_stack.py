import torch

x = torch.arange(6).reshape(2, 3)
y = torch.arange(6, 12).reshape(2, 3)

cat = torch.cat([x, y], dim=0)
print(f'cat (dim=0):\n{cat}')

stack = torch.stack([x, y], dim=1)
print(f'stack (dim=1):\n{stack}')

split = torch.split(x, 1, dim=0)
print(f'split: {[t.tolist() for t in split]}')

chunk = torch.chunk(x, 2, dim=1)
print(f'chunk: {[t.tolist() for t in chunk]}')

unbind = torch.unbind(x, dim=0)
print(f'unbind: {[t.tolist() for t in unbind]}') 