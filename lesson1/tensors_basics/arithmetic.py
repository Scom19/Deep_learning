import torch

# Арифметика с одинаковыми шейпами
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)
print(f'A = {A}')
print(f'B = {B}')
print(f'A + B = {A + B}')
print(f'A - B = {A - B}')
print(f'A * B = {A * B}')
print(f'A / B = {A / B}')
print(f'A ** 2 = {A ** 2}')
print(f'torch.pow(A, 3) = {torch.pow(A, 3)}')
print(f'torch.sqrt(B) = {torch.sqrt(B)}')
print(f'torch.exp(A) = {torch.exp(A)}')
print(f'torch.log(B) = {torch.log(B)}')

# Матричное умножение
O = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
P = torch.tensor([[2, 0], [0, 2]], dtype=torch.float32)
print(f'O @ P = {O @ P}')
print(f'torch.matmul(O, P) = {torch.matmul(O, P)}')

# Batch матричное умножение
Q = torch.randn(10, 3, 4)
R = torch.randn(10, 4, 5)
print(f'Q.shape = {Q.shape}, R.shape = {R.shape}')
print(f'batch matmul: torch.matmul(Q, R).shape = {torch.matmul(Q, R).shape}') 