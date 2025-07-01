import torch
import numpy as np

x = torch.arange(12)
print(f'x: {x}')

# view и reshape
x2 = x.view(3, 4)
print(f'view (3,4): {x2}')

x3 = x.reshape(2, 6)
print(f'reshape (2,6): {x3}')

# squeeze и unsqueeze
x4 = torch.zeros(1, 2, 1, 3)
print(f'x4: {x4.shape}')
print(f'squeeze: {x4.squeeze().shape}')
print(f'unsqueeze: {x4.unsqueeze(0).shape}')

# permute и transpose
x5 = torch.randn(2, 3, 4)
print(f'x5: {x5.shape}')
print(f'permute (2,0,1): {x5.permute(2, 0, 1).shape}')
print(f'transpose (1,2): {x5.transpose(1, 2).shape}')

# contiguous
x6 = x5.transpose(1, 2)
print(f'is contiguous: {x6.is_contiguous()}')
x6c = x6.contiguous()
print(f'contiguous: {x6c.is_contiguous()}')

# to (смена устройства/типа)
x7 = x2.to(dtype=torch.float32)
print(f'to float32: {x7}')

x8 = x7.to(torch.int64)
print(f'to int64: {x8}')

x9 = x7.type(torch.complex64)
print(f'type(torch.complex64): {x9}')

x10 = x7.float()
print(f'float(): {x10}')

x11 = x7.long()
print(f'long(): {x11}')

if torch.cuda.is_available():
    x_cuda = x7.to('cuda')
    print(f'to cuda: {x_cuda}')
    x_cpu = x_cuda.to('cpu')
    print(f'to cpu: {x_cpu}')

# numpy
x_np = x7.numpy()
print(f'numpy: {x_np}, {type(x_np)}')

# из numpy обратно в torch
y = torch.from_numpy(x_np)
print(f'from numpy: {y}') 