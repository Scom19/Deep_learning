import torch
import torch.nn as nn
import onnx
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from tabulate import tabulate
from core.torch_onnx import test_onnx_model_cuda_timer, convert_to_onnx
from core.torch_trt import test_torch_trt_model
from core.datasets import CustomImageDataset, RandomImageDataset
from core.utils import run_test, gpu_mem_usage
from core.model import Resnet18


def make_res_table(results: List[Dict], test_fn_names: Dict[str, str]):
    """
    Create a table for printing timing results.
    
    Args:
        results: List of result dictionaries with timing information
        test_fn_names: Dictionary mapping function names to display names
    """
    headers = [
        'Model',
        'Dataset',
        'Precision',
        'Timer Type',
        'Batch Size',
        'Avg Time (ms)',
        'Memory (MB)',
        'Speed (FPS)',
        'Speedup'
    ]
    table = []
    base_time = None
    for result in results:
        if base_time is None and result['timer_type'] == 'cuda' and result['precision'] == 'fp32':
            base_time = list(result['results'].values())[0]

    for result in results:
        for shape, time_res in result['results'].items():
            speedup = base_time / time_res if base_time is not None else 0
            fps = 1000 / time_res if time_res > 0 else 0
            table.append([
                test_fn_names.get(result['test_fn'], result['test_fn']),
                result['dataset'],
                result['precision'],
                result['timer_type'],
                shape[0],
                f"{time_res:.4f}",
                f"{result['memory']:.2f}",
                f"{fps:.2f}",
                f"{speedup:.2f}x"
            ])
    
    return tabulate(table, headers=headers, tablefmt='grid')


def test_torch_model(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_step: int = 1,
    num_runs: int = 50,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    precision: str = 'fp16',
    timer_type: str = 'cuda',
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    **kwargs
):
    """
    Test PyTorch model performance
    """
    model = model.cuda()
    
    if precision == 'fp16':
        model = model.half()
    
    model.eval()
    
    def model_wrapper(input_data):
        if precision == 'fp16':
            input_data = input_data.half()
        return model(input_data)
    
    results = run_test(
        model_wrapper=model_wrapper,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type=timer_type
    )
    
    return results


def test_onnx(
    model_path: str,
    dataset: Dataset,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 50,
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    precision: str = 'fp32',
    timer_type: str = 'cuda',
    **kwargs
):
    """
    Test ONNX model performance
    """
    onnx_path = model_path.replace('.pth', '.onnx')
    
    # Convert to ONNX if not exists
    if not torch.cuda.is_available():
        print("CUDA not available - skipping ONNX test")
        return {}
    
    convert_to_onnx(
        model_path=model_path,
        output_path=onnx_path,
        input_shape=input_shape,
        precision=precision,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size
    )
    
    # Always use CUDA timer
    return test_onnx_model_cuda_timer(
        onnx_path=onnx_path,
        input_shape=input_shape,
        batch_step=batch_step,
        dataset=dataset,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )


def test_torch_trt(
    model_path: str,
    dataset: Dataset,
    batch_step: int = 1,
    num_runs: int = 50,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    min_batch_size: int = 1,
    opt_batch_size: int = 1,
    max_batch_size: int = 1,
    precision: str = 'fp32',
    timer_type: str = 'cuda',
    **kwargs
):
    """
    Test Torch-TensorRT model performance
    """
    return test_torch_trt_model(
        model_path=model_path,
        input_shape=input_shape,
        precision=precision,
        batch_step=batch_step,
        dataset=dataset,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
        timer_type=timer_type
    )


def benchmark_models(
    model_path: str,
    num_runs: int = 50,
    min_batch_size: int = 64,
    max_batch_size: int = 64,
    opt_batch_size: int = 64,
    batch_step: int = 4,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
):
    """
    Benchmark all model types: PyTorch, ONNX, TensorRT
    """
    print(f"Benchmarking models from {model_path}")
    
    # Load model
    model = Resnet18(num_classes=6)  # 6 classes
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    # Extract image size from input_shape
    image_size = input_shape[1]  # Assuming square images
    
    # Create datasets
    try:
        real_dataset = CustomImageDataset(root_dir='./data/train', target_size=(image_size, image_size))
        print("Using real dataset")
    except:
        real_dataset = RandomImageDataset(target_size=input_shape)
        print("Using synthetic dataset")
    
    dummy_dataset = RandomImageDataset(target_size=input_shape)
    
    # Test configurations
    kwargs = {
        'datasets': [real_dataset, dummy_dataset],
        'precisions': ['fp16', 'fp32'],
        'timer_types': ['cuda']  # Only CUDA
    }

    test_functions = [
        test_torch_model,
        test_onnx,
        test_torch_trt
    ]
    
    test_fn_names = {
        'test_torch_model': 'PyTorch',
        'test_onnx': 'ONNX',
        'test_torch_trt': 'TensorRT'
    }
    
    static_kwargs = {
        'model_path': model_path,
        'model': model,
        'batch_step': batch_step,
        'num_runs': num_runs,
        'min_batch_size': min_batch_size,
        'max_batch_size': max_batch_size,
        'opt_batch_size': opt_batch_size,
        'input_shape': input_shape,
    }
    
    results = []
    
    for precision in kwargs['precisions']:
        for test_function in test_functions:
            for dataset in kwargs['datasets']:
                for timer_type in kwargs['timer_types']:
                    mem_usage = gpu_mem_usage  # Always use GPU memory
                    print(f'test params: test_function: {test_function.__name__}, dataloader: {dataset.__class__.__name__}, precision: {precision}, timer_type: {timer_type}')
                    result, allocated_memory = mem_usage(test_function)(
                        **static_kwargs,
                        dataset=dataset,
                        precision=precision,
                        timer_type=timer_type
                    )
                    
                    results.append({
                        'test_fn': test_function.__name__,
                        'dataset': dataset.__class__.__name__,
                        'precision': precision,
                        'timer_type': timer_type,
                        'results': result,
                        'memory': allocated_memory
                    })
    
    # Create and save report
    table = make_res_table(results, test_fn_names)
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(table)
    
    # Save to file
    with open('results.md', 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write("```\n")
        f.write(table)
        f.write("\n```\n")
    
    print(f"\nResults saved to results.md")
    
    # Convert results to format expected by benchmark_all.py
    formatted_results = {}
    
    # Group results by model type and get average times
    for result in results:
        model_name = test_fn_names.get(result['test_fn'], result['test_fn']).lower()
        
        if model_name not in formatted_results:
            formatted_results[model_name] = {}
        
        # Get average time from all batch sizes (take first result for now)
        if result['results']:
            avg_time = list(result['results'].values())[0]  # Take first batch size result
            formatted_results[model_name]['avg_time'] = avg_time
    
    return formatted_results

if __name__ == '__main__':
    benchmark_models(
        model_path='./weights/best_resnet18.pth',
        num_runs=50,
        min_batch_size=32,
        max_batch_size=64,
        opt_batch_size=64,
        batch_step=8,
        input_shape=(3, 224, 224),
    )
