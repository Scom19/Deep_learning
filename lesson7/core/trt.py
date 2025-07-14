import tensorrt as trt
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader
import pycuda.driver as cuda
import numpy as np
import os

try:
    from core.datasets import CustomImageDataset
    from core.utils import run_test
except ImportError:
    from datasets import CustomImageDataset
    from utils import run_test


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
RUNTIME = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path: str, engine_path: str, precision: str = 'fp16', input_shape: Tuple[int, int, int] = (3, 224, 224), min_batch_size: int = 1, max_batch_size: int = 1, opt_batch_size: int = 1) -> trt.ICudaEngine:
    # 1) Создаём builder, сеть и парсер
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # 2) Парсим ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Не удалось распарсить ONNX-модель")

        # 3) Создаём config и оптимизационный профиль
        config = builder.create_builder_config()

        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        # Имя входа берём из network.get_input(0).name
        input_name = network.get_input(0).name
        # Задаём min/opt/max shape для динамики: (N, C, H, W)
        min_shape = (min_batch_size, *input_shape)
        opt_shape = (opt_batch_size, *input_shape)
        max_shape = (max_batch_size, *input_shape)
        profile.set_shape(input_name,
                          min=min_shape,
                          opt=opt_shape,
                          max=max_shape)
        config.add_optimization_profile(profile)

        # 4) Строим движок сразу с config
        engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(engine)
    engine = RUNTIME.deserialize_cuda_engine(engine)
    return engine


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """
    Загружает сериализованный TensorRT-движок с диска.
    """
    with open(engine_path, 'rb') as f:
        engine = RUNTIME.deserialize_cuda_engine(f.read())
    return engine


def create_context(engine: trt.ICudaEngine, input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None) -> trt.IExecutionContext:
    context = engine.create_execution_context()
    if input_shapes:
        for name, shape in input_shapes.items():
            context.set_input_shape(name, shape)
    return context


def allocate_buffers(context: trt.IExecutionContext):
    engine = context.engine
    inputs: list = []
    outputs: list = []
    stream = cuda.Stream()

    for idx in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(idx)
        shape = tuple(context.get_tensor_shape(name))
        size  = int(np.prod(shape))
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Привязываем адрес GPU-буфера к тензору в контексте
        context.set_tensor_address(name, int(device_mem))

        entry = {'name': name, 'host': host_mem, 'device': device_mem}
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(entry)
        else:
            outputs.append(entry)

    return inputs, outputs, stream


def infer(context: trt.IExecutionContext, inputs, outputs, stream):
    # Асинхронная передача входов
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

    # Запуск инференса по новой API
    context.execute_async_v3(stream_handle=stream.handle)

    # Асинхронная передача выходов
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()

    return [out['host'] for out in outputs]


def test_pure_trt_model(
    engine: trt.ICudaEngine,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    dataloader: DataLoader = None
) -> Dict[Tuple[int, int, int], float]:
    """
    Тестирует чистую TensorRT модель с использованием run_test
    
    Args:
        engine: TensorRT engine
        input_shape: Форма входного тензора
        num_runs: Количество прогонов
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        dataloader: Даталоадер для тестирования
    
    Returns:
        Словарь с результатами тестирования
    """
    
    # Создаем обертку для TensorRT модели
    class TRTModelWrapper:
        def __init__(self, engine, input_shape):
            self.engine = engine
            self.input_shape = input_shape
            self.context = None
            self.inputs = None
            self.outputs = None
            self.stream = None
            self.current_batch_size = None

        def recreate_context(self, batch_size):
            if self.inputs is not None and self.outputs is not None:
                for buf in self.inputs + self.outputs:
                    buf['device'].free()
            self.context = self.engine.create_execution_context()
            self.context.set_input_shape('input', (batch_size, *self.input_shape))
            self.inputs, self.outputs, self.stream = allocate_buffers(self.context)
        
        def __call__(self, input_tensor, *args, **kwargs):
            batch_size = input_tensor.shape[0]
            
            # Пересоздаем контекст и аллоцируем буферы при смене размера батча
            if self.current_batch_size != batch_size:
                self.recreate_context(batch_size)
                self.current_batch_size = batch_size
            
            # Подготавливаем входные данные
            input_data = input_tensor.contiguous().view(-1).cpu().numpy()
            if self.inputs and len(self.inputs) > 0:
                np.copyto(self.inputs[0]['host'], input_data)
            
            # Выполняем инференс
            return infer(self.context, self.inputs, self.outputs, self.stream)
        
        def cleanup(self):
            """Безопасно освобождает ресурсы"""
            if self.inputs is not None and self.outputs is not None:
                try:
                    for buf in self.inputs + self.outputs:
                        buf['device'].free()
                except:
                    pass  # Игнорируем ошибки при освобождении ресурсов
    
    # Создаем обертку модели
    model_wrapper = TRTModelWrapper(engine, input_shape)
    
    try:
        # Запускаем тест через run_test
        return run_test(
            model_wrapper=model_wrapper,
            input_shape=input_shape,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            batch_step=1,
            dataset=dataloader.dataset if dataloader else None,
            timer_type='cuda'
        )
    finally:
        # Безопасно освобождаем ресурсы
        model_wrapper.cleanup()


if __name__ == '__main__':
    onnx_path = './weights/resnet18.onnx'
    pure_trt_path = './weights/resnet18_pure_trt.engine'

    # Инициализируем CUDA контекст
    cuda.init()
    ctx = cuda.Device(0).make_context()

    try:
        train_dataset = CustomImageDataset(root_dir='../lesson5_augmentations/data/train', target_size=(224, 224))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=8)
        
        # Конвертируем в чистый TensorRT
        if os.path.exists(onnx_path):
            engine = build_engine(onnx_path, pure_trt_path, precision='fp16', min_batch_size=32, max_batch_size=32, opt_batch_size=32)
            pure_trt_time = test_pure_trt_model(engine, dataloader=train_loader, min_batch_size=32, max_batch_size=32, input_shape=(3, 224, 224), num_runs=50)
            for shape, time in pure_trt_time.items():
                print(f"Shape: {shape}, Time: {time:.4f} ms")
    finally:
        # Освобождаем CUDA контекст
        ctx.pop()