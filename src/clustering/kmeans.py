"""
K-Means clustering para compressão de imagens.
Suporta GPU (CuPy) e CPU (NumPy) com processamento em batches.
"""

import gc
import numpy as np

# Tentar importar CuPy para GPU
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# ========== GERENCIAMENTO DE MEMÓRIA ==========

def clear_gpu_memory():
    """Limpa a memória da GPU liberando todos os blocos."""
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        gc.collect()


def get_array_module(use_gpu=True):
    """
    Retorna o módulo de arrays apropriado.
    
    Args:
        use_gpu: Se True e CuPy disponível, retorna cupy. Caso contrário, numpy.
        
    Returns:
        Módulo cupy ou numpy
    """
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_device(array, use_gpu=True):
    """
    Move array para o device correto (CPU ou GPU).
    
    Args:
        array: Array numpy ou cupy
        use_gpu: Se True, move para GPU (se disponível)
        
    Returns:
        Array no device correto
    """
    xp = get_array_module(use_gpu)
    
    if use_gpu and CUPY_AVAILABLE:
        if not isinstance(array, cp.ndarray):
            return cp.asarray(array)
        return array
    else:
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)


def to_cpu(array):
    """
    Move array para CPU (NumPy).
    
    Args:
        array: Array numpy ou cupy
        
    Returns:
        Array numpy
    """
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


# ========== DISTÂNCIA EUCLIDIANA ==========

def compute_euclidean_distances(X, centroids, xp, batch_size):
    """
    Calcula distâncias euclidianas entre pontos e centróides.
    Processa diretamente em batches sem criar array gigante.
    
    Args:
        X: Dados (n_samples, n_features)
        centroids: Centróides (n_centroids, n_features)
        xp: Módulo numpy ou cupy
        batch_size: Número de amostras por batch
        
    Returns:
        Array de índices dos centróides mais próximos
    """
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    
    # Verificar se array completo caberia na memória
    estimated_memory_gb = (n_samples * n_centroids * 4) / (1024 ** 3)
    
    if estimated_memory_gb > 10:
        # Array muito grande - processar batch por batch
        return _compute_distances_batched(X, centroids, xp, batch_size)
    else:
        # Array pequeno - processar tudo de uma vez (mais rápido)
        return _compute_distances_direct(X, centroids, xp)


def _compute_distances_direct(X, centroids, xp):
    """Calcula distâncias diretamente (para datasets pequenos)."""
    distances = xp.linalg.norm(X[:, xp.newaxis] - centroids, axis=2)
    idx = xp.argmin(distances, axis=1).astype(int)
    del distances
    return idx


def _compute_distances_batched(X, centroids, xp, batch_size):
    """
    Calcula distâncias em batches (para datasets grandes).
    Não aloca array completo - economiza memória.
    """
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    estimated_memory_gb = (n_samples * n_centroids * 4) / (1024 ** 3)
    
    print(f"      ⚠️  Array de distâncias seria {estimated_memory_gb:.1f} GB!")
    print(f"      💡 Processando índices diretamente em batches...")
    
    idx = xp.zeros(n_samples, dtype=int)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for batch_idx, start_idx in enumerate(range(0, n_samples, batch_size)):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        
        # Calcular distâncias APENAS para este batch
        batch_distances = xp.linalg.norm(
            batch[:, xp.newaxis] - centroids, 
            axis=2
        )
        
        # Encontrar índices do mínimo
        idx[start_idx:end_idx] = xp.argmin(batch_distances, axis=1).astype(int)
        
        # Limpar memória
        del batch_distances
        
        # Mostrar progresso
        if n_batches > 10 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
            progress = ((batch_idx + 1) / n_batches) * 100
            print(f"         Progresso: {progress:.0f}% "
                  f"({batch_idx + 1}/{n_batches} batches)")
        
        # Limpar memória GPU periodicamente
        if (batch_idx + 1) % 5 == 0 and xp == cp and CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
    
    return idx


# ========== K-MEANS CORE ==========

def find_closest_centroids(X, centroids, use_gpu=True, batch_size=200000):
    """
    Encontra o centróide mais próximo para cada ponto.
    
    Args:
        X: Dados (n_samples, n_features)
        centroids: Centróides (n_centroids, n_features)
        use_gpu: Usar GPU se disponível
        batch_size: Tamanho do batch para processamento
        
    Returns:
        Array de índices (n_samples,) indicando o centróide mais próximo
    """
    xp = get_array_module(use_gpu)
    
    X = to_device(X, use_gpu)
    centroids = to_device(centroids, use_gpu)
    
    idx = compute_euclidean_distances(X, centroids, xp, batch_size)
    
    # Limpar memória
    if use_gpu and CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    
    return idx


def compute_centroids(X, idx, K, use_gpu=True):
    """
    Calcula novos centróides como média dos pontos atribuídos.
    
    Args:
        X: Dados (n_samples, n_features)
        idx: Índices dos centróides atribuídos (n_samples,)
        K: Número de centróides
        use_gpu: Usar GPU se disponível
        
    Returns:
        Novos centróides (K, n_features)
    """
    xp = get_array_module(use_gpu)
    
    X = to_device(X, use_gpu)
    idx = to_device(idx, use_gpu)
    
    m, n = X.shape
    centroids = xp.zeros((K, n), dtype=X.dtype)

    for i in range(K):
        mask = (idx == i)
        points_assigned = X[mask]
        count = xp.sum(mask)
        
        if count > 0:
            centroids[i] = xp.mean(points_assigned, axis=0)
    
    return centroids


def kmeans_init_centroids(X, K, use_gpu=True):
    """
    Inicializa centróides aleatoriamente (Random Partition).
    
    Args:
        X: Dados (n_samples, n_features)
        K: Número de centróides
        use_gpu: Usar GPU se disponível
        
    Returns:
        Centróides iniciais (K, n_features)
    """
    xp = get_array_module(use_gpu)
    X = to_device(X, use_gpu)
    
    if use_gpu and CUPY_AVAILABLE:
        randidx = cp.random.permutation(X.shape[0])
    else:
        randidx = np.random.permutation(X.shape[0])
    
    centroids = X[randidx[:K]]
    return centroids


def run_kmeans(X, initial_centroids, max_iters=10, use_gpu=True, 
               batch_size=200000):
    """
    Executa algoritmo K-Means.
    
    Args:
        X: Dados (n_samples, n_features)
        initial_centroids: Centróides iniciais (K, n_features)
        max_iters: Número máximo de iterações
        use_gpu: Usar GPU se disponível
        batch_size: Tamanho do batch para processamento
        
    Returns:
        centroids: Centróides finais (K, n_features)
        idx: Atribuições finais (n_samples,)
    """
    xp = get_array_module(use_gpu)
    
    # Limpar memória antes de começar
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    # Mensagem de status
    device_name = "GPU" if use_gpu and CUPY_AVAILABLE else "CPU"
    print(f"🚀 Executando K-Means na {device_name} "
          f"(batch_size={batch_size:,})")
    
    # Preparar dados
    X = to_device(X, use_gpu)
    centroids = to_device(initial_centroids, use_gpu)
    
    m, n = X.shape
    K = centroids.shape[0]
    idx = xp.zeros(m, dtype=int)

    # Loop principal do K-Means
    for iteration in range(max_iters):
        print(f"K-Means iteration {iteration}/{max_iters - 1} "
              f"(device: {device_name})")
        
        # Passo 1: Atribuir pontos aos centróides mais próximos
        idx = find_closest_centroids(X, centroids, use_gpu, batch_size)
        
        # Passo 2: Recalcular centróides
        centroids = compute_centroids(X, idx, K, use_gpu)
        
        # Limpar memória periodicamente
        if use_gpu and CUPY_AVAILABLE and iteration % 2 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Retornar resultados na CPU
    centroids = to_cpu(centroids)
    idx = to_cpu(idx)
    
    # Limpar memória final
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    return centroids, idx