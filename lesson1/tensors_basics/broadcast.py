import torch

# Broadcasting: 2D + 1D
C = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
D = torch.tensor([10, 20, 30], dtype=torch.float32)
print(f'C = {C}')
print(f'D = {D}')
print(f'C + D (broadcast) = {C + D}')

# Broadcasting: 3D + 1D
E = torch.ones((2, 3, 4))
F = torch.arange(4)
print(f'E.shape = {E.shape}, F.shape = {F.shape}')
print(f'E + F (broadcast) = {E + F}')

# Broadcasting: 3D + 2D
G = torch.ones((2, 3, 4))
H = torch.arange(12).reshape(3, 4)
print(f'G.shape = {G.shape}, H.shape = {H.shape}')
print(f'G + H (broadcast) = {G + H}')

# Broadcasting: 4D + 1D
I = torch.ones((2, 3, 4, 5))
J = torch.arange(5)
print(f'I.shape = {I.shape}, J.shape = {J.shape}')
print(f'I + J (broadcast) = {I + J}')

# Broadcasting: 4D + 2D
K = torch.ones((2, 3, 4, 5))
L = torch.arange(20).reshape(4, 5)
print(f'K.shape = {K.shape}, L.shape = {L.shape}')
print(f'K + L (broadcast) = {K + L}')

# Broadcasting: 4D + 3D
M = torch.ones((2, 3, 4, 5))
N = torch.arange(60).reshape(3, 4, 5)
print(f'M.shape = {M.shape}, N.shape = {N.shape}')
print(f'M + N (broadcast) = {M + N}') 