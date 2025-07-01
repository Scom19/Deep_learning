import torch

# Autograd: вычисление градиентов для вектора
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1
z = y.sum()
print(f'x = {x}')
print(f'y = x**2 + 2x + 1 = {y}')
print(f'z = y.sum() = {z}')
z.backward()
print(f'x.grad = {x.grad}')

# Autograd для матриц
A = torch.randn(2, 2, requires_grad=True)
B = torch.randn(2, 2, requires_grad=True)
C = (A * B).sum()
print(f'A = {A}')
print(f'B = {B}')
print(f'C = (A * B).sum() = {C}')
C.backward()
print(f'A.grad = {A.grad}')
print(f'B.grad = {B.grad}') 