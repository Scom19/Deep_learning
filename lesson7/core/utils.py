import torch
import time
from typing import Callable, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import psutil
import subprocess
import threading

NUM_WARMUP_ITERATIONS = 10  # Уменьшено с 100 до 10 для ускорения

def get_gpu_info():
    """Получает информацию о видеокарте"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_props = torch.cuda.get_device_properties(0)
        memory_gb = device_props.total_memory / 1024**3
        return {
            'name': device_name,
            'memory_gb': memory_gb,
            'cuda_version': torch.version.cuda,
            'compute_capability': f"{device_props.major}.{device_props.minor}"
        }
    return None

def get_gpu_utilization():
    """Получает загруженность GPU через nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            gpu_util, mem_used, mem_total = map(int, line.split(', '))
            return {
                'gpu_utilization': gpu_util,
                'memory_used': mem_used,
                'memory_total': mem_total,
                'memory_util': (mem_used / mem_total) * 100
            }
    except:
        pass
    return None

class GPUMonitor:
    """Мониторинг GPU во время теста"""
    def __init__(self):
        self.monitoring = False
        self.utilization_history = []
        self.thread = None
    
    def start(self):
        """Начинает мониторинг"""
        self.monitoring = True
        self.utilization_history = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        """Останавливает мониторинг"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def _monitor(self):
        """Функция мониторинга в отдельном потоке"""
        while self.monitoring:
            gpu_info = get_gpu_utilization()
            if gpu_info:
                self.utilization_history.append(gpu_info)
            time.sleep(0.1)  # Проверяем каждые 100ms
    
    def get_stats(self):
        """Возвращает статистику загруженности"""
        if not self.utilization_history:
            return None
        
        gpu_utils = [info['gpu_utilization'] for info in self.utilization_history]
        return {
            'avg_gpu_util': np.mean(gpu_utils),
            'max_gpu_util': np.max(gpu_utils),
            'min_gpu_util': np.min(gpu_utils),
            'samples': len(gpu_utils)
        }

def cuda_timer(func):
    def wrapper(*args, **kwargs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream=torch.cuda.current_stream())
        result = func(*args, **kwargs)
        end_time.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return result, start_time.elapsed_time(end_time)
    return wrapper

def cpu_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() - start_time
        return result, end_time * 1000
    return wrapper

def gpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.max_memory_allocated()
        result = func(*args, **kwargs)
        return result, (torch.cuda.max_memory_allocated() - allocated_memory) / 2 ** 20
    return wrapper

def cpu_mem_usage(func):
    def wrapper(*args, **kwargs):
        allocated_memory = psutil.Process().memory_info().rss
        result = func(*args, **kwargs)
        return result, (psutil.Process().memory_info().rss - allocated_memory) / 2 ** 20
    return wrapper

def run_test(
    model_wrapper: Callable,
    data_preprocess: Callable = None,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    timer_type: str = 'cuda'
) -> Dict[str, any]:
    """
    Расширенная функция тестирования с GPU мониторингом и анализом времени на картинку
    
    Returns:
        Dict с результатами включая время на картинку, FPS, GPU utilization и анализ оптимального батча
    """
    shapes = [(size, *input_shape) for size in range(min_batch_size, max_batch_size + 1, batch_step)]
    results = {}
    timer = cuda_timer if timer_type == 'cuda' else cpu_timer
    device = 'cuda' if timer_type == 'cuda' else 'cpu'
    
    # Получаем информацию о GPU
    gpu_info = get_gpu_info()
    
    detailed_results = {
        'batch_results': {},
        'gpu_info': gpu_info,
        'per_image_times': {},
        'fps_values': {},
        'gpu_utilization': {},
        'optimal_batch_analysis': {}
    }
    
    for shape in shapes:
        batch_size = shape[0]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        # Инициализируем GPU мониторинг
        gpu_monitor = GPUMonitor() if timer_type == 'cuda' else None
        
        with torch.no_grad():
            # Warmup
            for _ in tqdm(range(NUM_WARMUP_ITERATIONS), desc=f'Warmup for batch {batch_size}'):
                dummy_input = torch.randn(shape, device=device)
                if data_preprocess:
                    dummy_input = data_preprocess(dummy_input)
                model_wrapper(dummy_input)
            
            # Начинаем мониторинг GPU
            if gpu_monitor:
                gpu_monitor.start()
            
            times = []
            for _ in range(num_runs):
                for batch in tqdm(dataloader, desc=f'Testing batch {batch_size}, iter {_}'):
                    image = batch[0].to(device)
                    if data_preprocess:
                        image = data_preprocess(image)
                    result, time_ms = timer(model_wrapper)(image)
                    times.append(time_ms)
            
            # Останавливаем мониторинг GPU
            if gpu_monitor:
                gpu_monitor.stop()
                gpu_stats = gpu_monitor.get_stats()
            else:
                gpu_stats = None
        
        # Обрабатываем результаты
        times = np.array(times)
        times = times[~np.isnan(times)]
        times = times[times < np.percentile(times, 90)]
        times = times[times > np.percentile(times, 10)]
        
        avg_time_per_batch = np.mean(times).item()
        avg_time_per_image = avg_time_per_batch / batch_size  # ВРЕМЯ НА КАРТИНКУ
        fps = 1000 / avg_time_per_batch  # FPS для батча
        fps_per_image = 1000 / avg_time_per_image  # FPS на картинку
        
        # Сохраняем результаты
        results[shape] = avg_time_per_batch  # Для совместимости с существующим кодом
        
        detailed_results['batch_results'][batch_size] = {
            'avg_time_per_batch_ms': avg_time_per_batch,
            'avg_time_per_image_ms': avg_time_per_image,
            'fps_batch': fps,
            'fps_per_image': fps_per_image,
            'std_time': np.std(times).item(),
            'samples': len(times)
        }
        
        detailed_results['per_image_times'][batch_size] = avg_time_per_image
        detailed_results['fps_values'][batch_size] = fps_per_image
        
        if gpu_stats:
            detailed_results['gpu_utilization'][batch_size] = gpu_stats
            
            # Анализ эффективности GPU
            efficiency = gpu_stats['avg_gpu_util'] / 100.0
            detailed_results['batch_results'][batch_size]['gpu_efficiency'] = efficiency
    
    # Анализ оптимального батча
    optimal_batch_analysis = analyze_optimal_batch(detailed_results)
    detailed_results['optimal_batch_analysis'] = optimal_batch_analysis
    
    return results, detailed_results

def analyze_optimal_batch(detailed_results: Dict) -> Dict:
    """
    Анализирует оптимальный размер батча на основе времени на картинку и загруженности GPU
    """
    if not detailed_results['per_image_times']:
        return {}
    
    batch_sizes = sorted(detailed_results['per_image_times'].keys())
    per_image_times = [detailed_results['per_image_times'][bs] for bs in batch_sizes]
    
    # Находим батч с минимальным временем на картинку
    min_time_idx = np.argmin(per_image_times)
    optimal_batch_by_time = batch_sizes[min_time_idx]
    
    # Анализ загруженности GPU
    gpu_utilization = detailed_results.get('gpu_utilization', {})
    optimal_batch_by_gpu = None
    
    if gpu_utilization:
        # Ищем батч с наибольшей загруженностью GPU (но не перегруженный)
        max_util = 0
        for batch_size in batch_sizes:
            if batch_size in gpu_utilization:
                avg_util = gpu_utilization[batch_size]['avg_gpu_util']
                if avg_util > max_util and avg_util < 98:  # Не перегружен
                    max_util = avg_util
                    optimal_batch_by_gpu = batch_size
    
    # Проверяем, недогружается ли GPU на малых батчах
    underutilized_batches = []
    if gpu_utilization:
        for batch_size in batch_sizes:
            if batch_size in gpu_utilization:
                avg_util = gpu_utilization[batch_size]['avg_gpu_util']
                if avg_util < 80:  # Недогружен
                    underutilized_batches.append(batch_size)
    
    return {
        'optimal_batch_by_time': optimal_batch_by_time,
        'optimal_batch_by_gpu': optimal_batch_by_gpu,
        'min_time_per_image_ms': per_image_times[min_time_idx],
        'underutilized_batches': underutilized_batches,
        'time_analysis': {
            'batch_sizes': batch_sizes,
            'per_image_times': per_image_times,
            'improvement_with_larger_batch': per_image_times[0] / per_image_times[-1] if len(per_image_times) > 1 else 1.0
        }
    }