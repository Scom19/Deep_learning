import torch

# 2.1 Простые вычисления с градиентами
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z
f.backward()
print('Градиенты:')
print('df/dx =', x.grad.item())
print('df/dy =', y.grad.item())
print('df/dz =', z.grad.item())

print('аналитически:')
print('df/dx =', 2*x.item() + 2*y.item()*z.item())
print('df/dy =', 2*y.item() + 2*x.item()*z.item())
print('df/dz =', 2*z.item() + 2*x.item()*y.item())

# 2.2 (MSE)
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y_pred = w * x + b
loss = mse_loss(y_pred, y_true)
loss.backward()
print('\nГрадиенты MSE:')
print('dL/dw =', w.grad.item())
print('dL/db =', b.grad.item())

# 2.3 Цепное правило
t = torch.tensor(1.5, requires_grad=True)
f = torch.sin(t**2 + 1)
f.backward()
print('\nГрадиент df/dx:', t.grad.item())
# Аналитически: df/dt = cos(t^2 + 1) * 2t
t2 = torch.tensor(1.5, requires_grad=True)
f2 = torch.sin(t2**2 + 1)
grad = torch.autograd.grad(f2, t2)[0]
print('Проверка', grad.item())