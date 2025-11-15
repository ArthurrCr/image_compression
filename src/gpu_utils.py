import numpy as np

def check_gpu():
    """
    Verifica se há GPU disponível e qual biblioteca usar.
    Retorna: (tem_gpu, biblioteca, device_info)
    """
    gpu_info = {
        'has_gpu': False,
        'library': 'numpy',
        'device': 'CPU',
        'gpu_name': None,
        'gpu_memory': None
    }
    
    # Tentar CuPy (CUDA/NVIDIA)
    try:
        import cupy as cp
        gpu_info['has_gpu'] = True
        gpu_info['library'] = 'cupy'
        gpu_info['device'] = 'GPU (CUDA)'
        
        # Informações da GPU
        device = cp.cuda.Device()
        gpu_info['gpu_name'] = device.compute_capability
        gpu_info['gpu_memory'] = f"{device.mem_info[1] / 1e9:.2f} GB"
        
        print("✅ GPU (CUDA) detectada via CuPy")
        print(f"   Memória: {gpu_info['gpu_memory']}")
        return gpu_info
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  CuPy instalado mas erro ao acessar GPU: {e}")
    
    # Tentar PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['has_gpu'] = True
            gpu_info['library'] = 'pytorch'
            gpu_info['device'] = 'GPU (CUDA via PyTorch)'
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            
            print("✅ GPU (CUDA) detectada via PyTorch")
            print(f"   GPU: {gpu_info['gpu_name']}")
            print(f"   Memória: {gpu_info['gpu_memory']}")
            return gpu_info
    except ImportError:
        pass
    
    # Nenhuma GPU encontrada
    print("ℹ️  Nenhuma GPU detectada, usando CPU")
    return gpu_info


def get_array_module(use_gpu=True):
    """
    Retorna o módulo apropriado (numpy ou cupy) baseado na disponibilidade de GPU.
    """
    if use_gpu:
        try:
            import cupy as cp
            print("🚀 Usando CuPy (GPU)")
            return cp
        except ImportError:
            print("⚠️  CuPy não instalado, usando NumPy (CPU)")
            return np
    else:
        print("💻 Usando NumPy (CPU)")
        return np


def to_cpu(array):
    """Converte array GPU para CPU (se necessário)"""
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)


def to_gpu(array):
    """Converte array CPU para GPU (se disponível)"""
    try:
        import cupy as cp
        if not isinstance(array, cp.ndarray):
            return cp.asarray(array)
        return array
    except ImportError:
        return array