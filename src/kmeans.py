"""K-Means clustering para compressão de imagens com suporte GPU/CPU."""

import gc
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def clear_gpu_memory():
    """Limpa memória da GPU."""
    if CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


def get_module(use_gpu=True):
    """Retorna cupy se GPU disponível, senão numpy."""
    return cp if (use_gpu and CUPY_AVAILABLE) else np


def to_device(array, use_gpu=True):
    """Move array para GPU ou CPU."""
    if use_gpu and CUPY_AVAILABLE:
        return cp.asarray(array)
    return np.asarray(cp.asnumpy(array) if isinstance(array, cp.ndarray)
                      else array)


def compute_distances(X, centroids, xp, batch_size):
    """
    Calcula distâncias euclidianas entre pontos e centróides.
    
    Processa diretamente ou em batches dependendo do tamanho.
    """
    n_samples, n_centroids = X.shape[0], centroids.shape[0]
    memory_gb = (n_samples * n_centroids * 4) / (1024 ** 3)
    
    if memory_gb > 10:
        return _compute_batched(X, centroids, xp, batch_size, memory_gb)
    
    distances = xp.linalg.norm(X[:, xp.newaxis] - centroids, axis=2)
    idx = xp.argmin(distances, axis=1).astype(int)
    del distances
    return idx


def _compute_batched(X, centroids, xp, batch_size, memory_gb):
    """Calcula distâncias em batches para economizar memória."""
    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    
    print(f"      ⚠️  Array seria {memory_gb:.1f} GB!")
    print(f"      💡 Processando em {n_batches} batches...")
    
    idx = xp.zeros(n_samples, dtype=int)
    
    for i, start in enumerate(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch = X[start:end]
        
        batch_dist = xp.linalg.norm(
            batch[:, xp.newaxis] - centroids, axis=2
        )
        idx[start:end] = xp.argmin(batch_dist, axis=1).astype(int)
        del batch_dist
        
        if n_batches > 10 and (i + 1) % max(1, n_batches // 10) == 0:
            print(f"         {(i + 1) / n_batches * 100:.0f}% "
                  f"({i + 1}/{n_batches})")
        
        if (i + 1) % 5 == 0 and xp == cp and CUPY_AVAILABLE:
            clear_gpu_memory()
    
    return idx


def find_closest_centroids(X, centroids, use_gpu=True, batch_size=200000):
    """Encontra centróide mais próximo para cada ponto."""
    xp = get_module(use_gpu)
    X, centroids = to_device(X, use_gpu), to_device(centroids, use_gpu)
    
    idx = compute_distances(X, centroids, xp, batch_size)
    
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    return idx


def compute_centroids(X, idx, K, use_gpu=True):
    """Calcula novos centróides como média dos pontos atribuídos."""
    xp = get_module(use_gpu)
    X, idx = to_device(X, use_gpu), to_device(idx, use_gpu)
    
    centroids = xp.zeros((K, X.shape[1]), dtype=X.dtype)
    
    for i in range(K):
        mask = (idx == i)
        if xp.sum(mask) > 0:
            centroids[i] = xp.mean(X[mask], axis=0)
    
    return centroids


def kmeans_init_centroids(X, K, use_gpu=True):
    """Inicializa centróides aleatoriamente."""
    xp = get_module(use_gpu)
    X = to_device(X, use_gpu)
    
    randidx = xp.random.permutation(X.shape[0])
    return X[randidx[:K]]


def run_kmeans(X, initial_centroids, max_iters=10, use_gpu=True,
               batch_size=200000):
    """
    Executa algoritmo K-Means.
    
    Args:
        X: Dados (n_samples, n_features)
        initial_centroids: Centróides iniciais (K, n_features)
        max_iters: Número máximo de iterações
        use_gpu: Usar GPU se disponível
        batch_size: Tamanho do batch
        
    Returns:
        centroids: Centróides finais
        idx: Atribuições finais
    """
    xp = get_module(use_gpu)
    device = "GPU" if (use_gpu and CUPY_AVAILABLE) else "CPU"
    
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    print(f"🚀 K-Means na {device} (batch_size={batch_size:,})")
    
    X = to_device(X, use_gpu)
    centroids = to_device(initial_centroids, use_gpu)
    K = centroids.shape[0]
    
    for i in range(max_iters):
        print(f"Iteração {i}/{max_iters - 1} ({device})")
        
        idx = find_closest_centroids(X, centroids, use_gpu, batch_size)
        centroids = compute_centroids(X, idx, K, use_gpu)
        
        if use_gpu and CUPY_AVAILABLE and i % 2 == 0:
            clear_gpu_memory()
    
    # Retorna resultados na CPU
    centroids = np.asarray(cp.asnumpy(centroids) if isinstance(
        centroids, cp.ndarray) else centroids)
    idx = np.asarray(cp.asnumpy(idx) if isinstance(idx, cp.ndarray) else idx)
    
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    return centroids, idx