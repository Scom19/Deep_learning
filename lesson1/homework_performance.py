import torch
import time

# 3.1
shapes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
matrices = [torch.rand(shape) for shape in shapes]

if not torch.cuda.is_available():
    print('Внимание: CUDA (GPU) не обнаружена, сравнение только для CPU.')

# 3.2 Функция измерения времени
def measure_time_cpu(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return (end - start) * 1000, result

def measure_time_gpu(func, *args):
    # Создаем CUDA-события для измерения времени
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Синхронизируем устройство перед началом измерения чтобы все предыдущие операции завершились
    torch.cuda.synchronize()
    start.record()# Запускаем отсчет времени
    result = func(*args)
    end.record()# Останавливаем отсчет времени
    # Синхронизируем устройство, чтобы убедиться, что все операции завершены
    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)
    return time_ms, result

# 3.3 Сравнение операций

def run_benchmark():
    device_cpu = torch.device('cpu')
    device_gpu = torch.device('cuda')
    operations = [
        ("Матричное умножение", lambda x: torch.matmul(x, x.transpose(-1, -2))),
        ("Сложение", lambda x: x + x),
        ("Поэлементное умножение", lambda x: x * x),
        ("Транспонирование", lambda x: x.transpose(-1, -2)),
        ("Сумма элементов", lambda x: x.sum()),
    ]
    for shape, mat in zip(shapes, matrices):
        print(f"\nРазмер: {shape}")
        print(f"{'Операция':<22}| {'CPU (мс)':<10}| {'GPU (мс)':<10}| {'Ускорение'}")
        print('-' * 60)

        for name, op in operations:
            cpu_time, _ = measure_time_cpu(op, mat.to(device_cpu))
            mat_gpu = mat.to(device_gpu)
            gpu_time, _ = measure_time_gpu(op, mat_gpu)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"{name:<22}| {cpu_time:>9.2f} | {gpu_time:>9.2f} | {speedup:>9.2f}x")

run_benchmark()