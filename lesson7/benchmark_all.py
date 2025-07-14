import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np
import logging
from datetime import datetime

from core.compare import benchmark_models


def setup_logger(name='benchmark'):
    """Настраивает логгер для бенчмарка"""
    # Создаем папку для логов
    os.makedirs('./logs', exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики
    logger.handlers.clear()
    
    # Создаем файл для логов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'./logs/{name}_{timestamp}.log'
    
    # Создаем обработчик файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Создаем обработчик консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Создаем форматтер
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def run_benchmark_for_size(image_size: int, batch_sizes: List[int] = [1, 4, 16, 64], logger=None):
    """
    Тестирует производительность модели для указанного размера изображения
    """
    if logger is None:
        logger = setup_logger('benchmark_size')
    
    logger.info(f"\nБенчмарк для размера изображения {image_size}x{image_size}")
    
    model_path = f'./weights/best_resnet18_{image_size}.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Модель для размера {image_size} не найдена: {model_path}")
        return None
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\nТестирование размера батча: {batch_size}")
        
        try:
            batch_results = benchmark_models(
                model_path=model_path,
                num_runs=3,
                min_batch_size=batch_size,
                max_batch_size=batch_size,
                opt_batch_size=batch_size,
                batch_step=1,
                input_shape=(3, image_size, image_size)
            )
            
            if batch_results:
                results[batch_size] = batch_results
                logger.info(f"\nБенчмарк для батча {batch_size} завершен")
            else:
                logger.error(f"\nОшибка при бенчмарке для батча {batch_size}")
                
        except Exception as e:
            logger.error(f"\nОшибка при бенчмарке для батча {batch_size}: {e}")
    
    return results


def create_performance_plots(all_results: Dict[int, Dict], logger=None):
    """
    Создает и сохраняет графики производительности
    """
    if logger is None:
        logger = setup_logger('plots')
    
    # Создаем папку для графиков
    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info("Создание графиков производительности...")
    
    # Настройка стиля графиков
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # График 1: FPS vs Размер изображения
    logger.info("Создание графика: FPS vs Размер изображения")
    
    image_sizes = sorted(all_results.keys())
    methods = ['pytorch', 'onnx', 'tensorrt']
    method_labels = {'pytorch': 'PyTorch', 'onnx': 'ONNX', 'tensorrt': 'Torch-TensorRT'}
    colors = {'pytorch': 'blue', 'onnx': 'orange', 'tensorrt': 'green'}
    
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        fps_values = []
        valid_sizes = []
        
        for size in image_sizes:
            if size in all_results and all_results[size]:
                # Берем средний FPS по всем размерам батчей для каждого размера изображения
                batch_fps = []
                for batch_size, batch_results in all_results[size].items():
                    if method in batch_results and 'avg_time' in batch_results[method]:
                        avg_time = batch_results[method]['avg_time']
                        if avg_time > 0:
                            fps = 1000 / avg_time  # Конвертируем из мс в FPS
                            batch_fps.append(fps)
                
                if batch_fps:
                    avg_fps = np.mean(batch_fps)
                    fps_values.append(avg_fps)
                    valid_sizes.append(size)
        
        if fps_values:
            plt.plot(valid_sizes, fps_values, marker='o', linewidth=2, markersize=8,
                    label=method_labels[method], color=colors[method])
    
    plt.xlabel('Размер изображения (пиксели)', fontsize=14)
    plt.ylabel('FPS (кадров в секунду)', fontsize=14)
    plt.title('Производительность: FPS vs Размер изображения', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'fps_vs_image_size.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"График сохранен: {plot_path}")
    
    # График 2: FPS vs Размер батча (для размера изображения 224x224)
    logger.info("Создание графика: FPS vs Размер батча")
    
    reference_size = 224  # Используем 224x224 как референсный размер
    if reference_size in all_results and all_results[reference_size]:
        plt.figure(figsize=(12, 8))
        
        batch_sizes = sorted(all_results[reference_size].keys())
        
        for method in methods:
            fps_values = []
            valid_batches = []
            
            for batch_size in batch_sizes:
                if batch_size in all_results[reference_size]:
                    batch_results = all_results[reference_size][batch_size]
                    if method in batch_results and 'avg_time' in batch_results[method]:
                        avg_time = batch_results[method]['avg_time']
                        if avg_time > 0:
                            fps = 1000 / avg_time
                            fps_values.append(fps)
                            valid_batches.append(batch_size)
            
            if fps_values:
                plt.plot(valid_batches, fps_values, marker='o', linewidth=2, markersize=8,
                        label=method_labels[method], color=colors[method])
        
        plt.xlabel('Размер батча', fontsize=14)
        plt.ylabel('FPS (кадров в секунду)', fontsize=14)
        plt.title(f'Производительность: FPS vs Размер батча (изображения {reference_size}x{reference_size})', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, 'fps_vs_batch_size.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"График сохранен: {plot_path}")
    
    # График 3: Ускорение относительно PyTorch
    logger.info("Создание графика: Ускорение относительно PyTorch")
    
    plt.figure(figsize=(12, 8))
    
    optimization_methods = ['onnx', 'tensorrt']
    opt_labels = {'onnx': 'ONNX', 'tensorrt': 'Torch-TensorRT'}
    opt_colors = {'onnx': 'orange', 'tensorrt': 'green'}
    
    for method in optimization_methods:
        speedup_values = []
        valid_sizes = []
        
        for size in image_sizes:
            if size in all_results and all_results[size]:
                # Вычисляем среднее ускорение по всем размерам батчей
                speedups = []
                for batch_size, batch_results in all_results[size].items():
                    if ('pytorch' in batch_results and method in batch_results and 
                        'avg_time' in batch_results['pytorch'] and 'avg_time' in batch_results[method]):
                        
                        pytorch_time = batch_results['pytorch']['avg_time']
                        method_time = batch_results[method]['avg_time']
                        
                        if method_time > 0 and pytorch_time > 0:
                            speedup = pytorch_time / method_time
                            speedups.append(speedup)
                
                if speedups:
                    avg_speedup = np.mean(speedups)
                    speedup_values.append(avg_speedup)
                    valid_sizes.append(size)
        
        if speedup_values:
            plt.plot(valid_sizes, speedup_values, marker='o', linewidth=2, markersize=8,
                    label=opt_labels[method], color=opt_colors[method])
    
    plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='PyTorch (базовая линия)')
    plt.xlabel('Размер изображения (пиксели)', fontsize=14)
    plt.ylabel('Ускорение (раз)', fontsize=14)
    plt.title('Ускорение оптимизированных методов относительно PyTorch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, 'speedup_vs_pytorch.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"График сохранен: {plot_path}")
    
    # Дополнительный график: Ускорение vs Размер батча
    if reference_size in all_results and all_results[reference_size]:
        logger.info("Создание графика: Ускорение vs Размер батча")
        
        plt.figure(figsize=(12, 8))
        
        batch_sizes = sorted(all_results[reference_size].keys())
        
        for method in optimization_methods:
            speedup_values = []
            valid_batches = []
            
            for batch_size in batch_sizes:
                if batch_size in all_results[reference_size]:
                    batch_results = all_results[reference_size][batch_size]
                    if ('pytorch' in batch_results and method in batch_results and 
                        'avg_time' in batch_results['pytorch'] and 'avg_time' in batch_results[method]):
                        
                        pytorch_time = batch_results['pytorch']['avg_time']
                        method_time = batch_results[method]['avg_time']
                        
                        if method_time > 0 and pytorch_time > 0:
                            speedup = pytorch_time / method_time
                            speedup_values.append(speedup)
                            valid_batches.append(batch_size)
            
            if speedup_values:
                plt.plot(valid_batches, speedup_values, marker='o', linewidth=2, markersize=8,
                        label=opt_labels[method], color=opt_colors[method])
        
        plt.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='PyTorch (базовая линия)')
        plt.xlabel('Размер батча', fontsize=14)
        plt.ylabel('Ускорение (раз)', fontsize=14)
        plt.title(f'Ускорение vs Размер батча (изображения {reference_size}x{reference_size})', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, 'speedup_vs_batch_size.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"График сохранен: {plot_path}")
    
    logger.info(f"Все графики созданы и сохранены в папке: {plots_dir}")


def create_summary_report(all_results: Dict[int, Dict], logger=None):
    """
    Формирует сводный отчет с результатами тестирования
    """
    if logger is None:
        logger = setup_logger('benchmark_report')
    
    logger.info("\n" + "="*60)
    logger.info("СОЗДАНИЕ СВОДНОГО ОТЧЕТА")
    logger.info("="*60)
    
    # Создаем графики
    create_performance_plots(all_results, logger)
    
    with open('benchmark_report.md', 'w', encoding='utf-8') as f:
        f.write("# Отчет о бенчмарке производительности\n\n")
        
        # Информация о тестировании
        f.write("## Параметры тестирования\n")
        f.write("- Размеры изображений: 224x224, 256x256, 384x384, 512x512\n")
        f.write("- Размеры батчей: 1, 2, 4, 8, 16, 32, 64\n")
        f.write("- Подходы: PyTorch, ONNX Runtime, Torch-TensorRT\n")
        f.write("- Количество прогонов: 50\n\n")
        
        # Информация о видеокарте
        f.write("## Информация о видеокарте\n")
        f.write(f"- Видеокарта: {torch.cuda.get_device_name(0)}\n")
        f.write(f"- Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
        f.write(f"- CUDA версия: {torch.version.cuda}\n\n")
        
        # Добавляем ссылки на графики
        f.write("## Графики производительности\n\n")
        f.write("### График 1: FPS vs Размер изображения\n")
        f.write("![FPS vs Размер изображения](./plots/fps_vs_image_size.png)\n\n")
        
        f.write("### График 2: FPS vs Размер батча\n")
        f.write("![FPS vs Размер батча](./plots/fps_vs_batch_size.png)\n\n")
        
        f.write("### График 3: Ускорение относительно PyTorch\n")
        f.write("![Ускорение относительно PyTorch](./plots/speedup_vs_pytorch.png)\n\n")
        
        f.write("### График 4: Ускорение vs Размер батча\n")
        f.write("![Ускорение vs Размер батча](./plots/speedup_vs_batch_size.png)\n\n")
        
        # Результаты по каждому размеру
        for image_size, size_results in all_results.items():
            if size_results:
                f.write(f"## Результаты для размера {image_size}x{image_size}\n\n")
                
                # Создаем таблицу результатов
                f.write("| Размер батча | PyTorch (мс) | ONNX (мс) | TensorRT (мс) | Ускорение ONNX | Ускорение TRT |\n")
                f.write("|--------------|--------------|-----------|---------------|----------------|---------------|\n")
                
                for batch_size, batch_results in size_results.items():
                    pytorch_time = batch_results.get('pytorch', {}).get('avg_time', 0)
                    onnx_time = batch_results.get('onnx', {}).get('avg_time', 0)
                    trt_time = batch_results.get('tensorrt', {}).get('avg_time', 0)
                    
                    onnx_speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
                    trt_speedup = pytorch_time / trt_time if trt_time > 0 else 0
                    
                    f.write(f"| {batch_size} | {pytorch_time:.2f} | {onnx_time:.2f} | {trt_time:.2f} | {onnx_speedup:.2f}x | {trt_speedup:.2f}x |\n")
                
                f.write("\n")
    
    logger.info("\nСводный отчет создан: benchmark_report.md")


def main():
    logger = setup_logger('benchmark_main')
    
    image_sizes = [224, 256, 384, 512]
    batch_sizes = [1, 4, 16, 64]
    
    logger.info(f"Размеры изображений: {image_sizes}")
    logger.info(f"Размеры батчей: {batch_sizes}")
    
    all_results = {}
    
    for image_size in image_sizes:
        results = run_benchmark_for_size(image_size, batch_sizes, logger)
        if results:
            all_results[image_size] = results
    
    # Создаем сводный отчет
    if all_results:
        create_summary_report(all_results, logger)
    else:
        logger.warning("Нет результатов для создания отчета")


if __name__ == '__main__':
    main() 