import torch

# 1.1 Создание тензоров
A = torch.rand(3, 4)
print('Тензор 3x4, случайные числа от 0 до 1:\n', A)

B = torch.zeros(2, 3, 4)
print('Тензор 2x3x4, заполненный нулями:\n', B)

C = torch.ones(5, 5)
print('Тензор 5x5, заполненный единицами:\n', C)

D = torch.arange(16).reshape(4, 4)
print('Тензор 4x4 с числами от 0 до 15:\n', D)

# 1.2 Операции с тензорами
A = torch.rand(3, 4)
B = torch.rand(4, 3)

# Транспонирование тензора A
A_T = A.t()
print('A.T =\n', A_T)

# Матричное умножение A и B
matmul = torch.matmul(A, B)
print('A @ B =\n', matmul)

# Поэлементное умножение A и транспонированного B
B_T = B.t()
elem_mul = A * B_T
print('\nA * B.T =\n', elem_mul)

sum_A = A.sum()
print('Сумма всех элементов A:', sum_A.item())

# 1.3 Индексация и срезы
T = torch.arange(125).reshape(5, 5, 5)

first_row = T[0, 0, :]
print('Первая строка', first_row)

last_col = T[0, :, -1]
print('Последний столбец', last_col)

center = T[0, 2:4, 2:4]
print('Подматрица 2x2 из центра:', center)

even_idx = T[0, ::2, ::2]
print('Элементы с четными индексами:', even_idx)

# 1.4 Работа с формами
flat = torch.arange(24)
print('\nТензор из 24 элементов:', flat)
print('2x12:\n', flat.reshape(2, 12))
print('3x8:\n', flat.reshape(3, 8))
print('4x6:\n', flat.reshape(4, 6))
print('2x3x4:\n', flat.reshape(2, 3, 4))
print('2x2x2x3:\n', flat.reshape(2, 2, 2, 3)) 