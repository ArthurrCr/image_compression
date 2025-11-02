import numpy as np
from matplotlib import colors as mcolors
import gc

# Tentar importar CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def clear_gpu_memory():
    """Limpa a memória da GPU"""
    if CUPY_AVAILABLE:
        # Limpar cache da mempool
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        # Forçar garbage collection
        gc.collect()


def get_array_module(use_gpu=True):
    """Retorna numpy ou cupy baseado em disponibilidade"""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


def to_device(array, use_gpu=True):
    """Converte array para o device correto (CPU ou GPU)"""
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
    """Força conversão para CPU (NumPy)"""
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


# ========== FUNÇÕES DE DISTÂNCIA COM PROCESSAMENTO EM BATCHES ==========

def euclidean_distance(X, centroids, xp=np, batch_size=50000):
    """
    Distância Euclidiana com processamento em batches para economizar memória.
    """
    if xp == cp and CUPY_AVAILABLE:
        X = cp.asarray(X)
        centroids = cp.asarray(centroids)
    else:
        X = np.asarray(X)
        centroids = np.asarray(centroids)
    
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    
    # Se dataset pequeno, processar tudo de uma vez
    if n_samples <= batch_size:
        return xp.linalg.norm(X[:, xp.newaxis] - centroids, axis=2)
    
    # Processar em batches
    distances = xp.zeros((n_samples, n_centroids), dtype=X.dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        distances[start_idx:end_idx] = xp.linalg.norm(
            batch[:, xp.newaxis] - centroids, axis=2
        )
    
    return distances


def manhattan_distance(X, centroids, xp=np, batch_size=50000):
    """Distância Manhattan com batches"""
    if xp == cp and CUPY_AVAILABLE:
        X = cp.asarray(X)
        centroids = cp.asarray(centroids)
    else:
        X = np.asarray(X)
        centroids = np.asarray(centroids)
    
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    
    if n_samples <= batch_size:
        return xp.sum(xp.abs(X[:, xp.newaxis] - centroids), axis=2)
    
    distances = xp.zeros((n_samples, n_centroids), dtype=X.dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        distances[start_idx:end_idx] = xp.sum(
            xp.abs(batch[:, xp.newaxis] - centroids), axis=2
        )
    
    return distances


def cosine_distance(X, centroids, xp=np, batch_size=50000):
    """Distância Cosseno com batches"""
    if xp == cp and CUPY_AVAILABLE:
        X = cp.asarray(X)
        centroids = cp.asarray(centroids)
    else:
        X = np.asarray(X)
        centroids = np.asarray(centroids)
    
    X_norm = X / (xp.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (xp.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    
    n_samples = X_norm.shape[0]
    
    if n_samples <= batch_size:
        similarity = xp.dot(X_norm, C_norm.T)
        return 1 - similarity
    
    # Processar em batches
    similarities = xp.zeros((n_samples, centroids.shape[0]), dtype=X.dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X_norm[start_idx:end_idx]
        similarities[start_idx:end_idx] = xp.dot(batch, C_norm.T)
    
    return 1 - similarities


def chebyshev_distance(X, centroids, xp=np, batch_size=50000):
    """Distância Chebyshev com batches"""
    if xp == cp and CUPY_AVAILABLE:
        X = cp.asarray(X)
        centroids = cp.asarray(centroids)
    else:
        X = np.asarray(X)
        centroids = np.asarray(centroids)
    
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    
    if n_samples <= batch_size:
        return xp.max(xp.abs(X[:, xp.newaxis] - centroids), axis=2)
    
    distances = xp.zeros((n_samples, n_centroids), dtype=X.dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        distances[start_idx:end_idx] = xp.max(
            xp.abs(batch[:, xp.newaxis] - centroids), axis=2
        )
    
    return distances


def minkowski_distance(X, centroids, p=3, xp=np, batch_size=50000):
    """Distância Minkowski com batches"""
    if xp == cp and CUPY_AVAILABLE:
        X = cp.asarray(X)
        centroids = cp.asarray(centroids)
    else:
        X = np.asarray(X)
        centroids = np.asarray(centroids)
    
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    
    if n_samples <= batch_size:
        return xp.sum(xp.abs(X[:, xp.newaxis] - centroids) ** p, axis=2) ** (1/p)
    
    distances = xp.zeros((n_samples, n_centroids), dtype=X.dtype)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        distances[start_idx:end_idx] = xp.sum(
            xp.abs(batch[:, xp.newaxis] - centroids) ** p, axis=2
        ) ** (1/p)
    
    return distances


DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'chebyshev': chebyshev_distance,
    'minkowski': minkowski_distance,
}


# ========== CONVERSÕES (mantém igual) ==========

def rgb_to_hsv_vectorized(rgb_array):
    """Converte RGB para HSV"""
    rgb_cpu = to_cpu(rgb_array)
    
    if rgb_cpu.ndim == 2:
        h, w = 1, rgb_cpu.shape[0]
        rgb_img = rgb_cpu.reshape(h, w, 3)
        hsv_img = mcolors.rgb_to_hsv(rgb_img)
        result = hsv_img.reshape(-1, 3)
    else:
        result = mcolors.rgb_to_hsv(rgb_cpu)
    
    return result


def hsv_to_rgb_vectorized(hsv_array):
    """Converte HSV para RGB"""
    hsv_cpu = to_cpu(hsv_array)
    
    if hsv_cpu.ndim == 2:
        h, w = 1, hsv_cpu.shape[0]
        hsv_img = hsv_cpu.reshape(h, w, 3)
        rgb_img = mcolors.hsv_to_rgb(hsv_img)
        result = rgb_img.reshape(-1, 3)
    else:
        result = mcolors.hsv_to_rgb(hsv_cpu)
    
    return result


# ========== K-MEANS ==========

def find_closest_centroids(X, centroids, distance_metric='euclidean', use_gpu=True):
    """Encontra centróide mais próximo com gerenciamento de memória."""
    xp = get_array_module(use_gpu)
    
    X = to_device(X, use_gpu)
    centroids = to_device(centroids, use_gpu)
    
    if isinstance(distance_metric, str):
        if distance_metric not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Métrica '{distance_metric}' não reconhecida.")
        distance_func = DISTANCE_FUNCTIONS[distance_metric]
    else:
        distance_func = distance_metric
    
    # Calcular distâncias com batching automático
    distances = distance_func(X, centroids, xp=xp)
    
    idx = xp.argmin(distances, axis=1).astype(int)
    
    # Limpar variável temporária grande
    del distances
    if use_gpu and CUPY_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    
    return idx


def compute_centroids(X, idx, K, use_gpu=True):
    """Calcula centróides."""
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


def kMeans_init_centroids(X, K, use_gpu=True):
    """Inicializa centróides."""
    xp = get_array_module(use_gpu)
    
    X = to_device(X, use_gpu)
    
    if use_gpu and CUPY_AVAILABLE:
        randidx = cp.random.permutation(X.shape[0])
    else:
        randidx = np.random.permutation(X.shape[0])
    
    centroids = X[randidx[:K]]
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False, 
               plot_function=None, distance_metric='euclidean', 
               color_space='rgb', use_gpu=True):
    """Executa K-Means com gerenciamento de memória."""
    xp = get_array_module(use_gpu)
    
    # Limpar memória antes de começar
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    if use_gpu and CUPY_AVAILABLE:
        print(f"🚀 Executando K-Means na GPU")
    else:
        print(f"💻 Executando K-Means na CPU")
    
    X = to_device(X, use_gpu)
    initial_centroids = to_device(initial_centroids, use_gpu)
    
    # Conversão de espaço de cor
    if color_space == 'hsv':
        print(f"🎨 Convertendo RGB → HSV")
        X_transformed = rgb_to_hsv_vectorized(X)
        initial_centroids_transformed = rgb_to_hsv_vectorized(initial_centroids)
        X_transformed = to_device(X_transformed, use_gpu)
        initial_centroids_transformed = to_device(initial_centroids_transformed, use_gpu)
    elif color_space == 'hls':
        print(f"⚠️  HLS não suportado ainda, usando RGB")
        X_transformed = X
        initial_centroids_transformed = initial_centroids
    else:
        X_transformed = X
        initial_centroids_transformed = initial_centroids
    
    m, n = X_transformed.shape
    K = initial_centroids_transformed.shape[0]
    centroids = initial_centroids_transformed.copy()
    previous_centroids = centroids.copy()
    idx = xp.zeros(m, dtype=int)

    for i in range(max_iters):
        print(f"K-Means iteration {i}/{max_iters-1} (espaço: {color_space.upper()}, métrica: {distance_metric}, device: {'GPU' if use_gpu and CUPY_AVAILABLE else 'CPU'})")
        
        idx = find_closest_centroids(X_transformed, centroids, 
                                     distance_metric=distance_metric, 
                                     use_gpu=use_gpu)
        
        if plot_progress and plot_function is not None:
            X_plot = to_cpu(X_transformed)
            centroids_plot = to_cpu(centroids)
            prev_plot = to_cpu(previous_centroids)
            idx_plot = to_cpu(idx)
            
            plot_function(X_plot, centroids_plot, prev_plot, idx_plot, K, i)
            previous_centroids = centroids.copy()
        
        centroids = compute_centroids(X_transformed, idx, K, use_gpu=use_gpu)
        
        # Limpar memória a cada iteração
        if use_gpu and CUPY_AVAILABLE and i % 2 == 0:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Converter de volta
    if color_space == 'hsv':
        centroids_rgb = hsv_to_rgb_vectorized(centroids)
        centroids_rgb = to_device(centroids_rgb, use_gpu)
    else:
        centroids_rgb = centroids
    
    # Retornar na CPU
    centroids_rgb = to_cpu(centroids_rgb)
    idx = to_cpu(idx)
    
    # Limpar memória GPU no final
    if use_gpu and CUPY_AVAILABLE:
        clear_gpu_memory()
    
    return centroids_rgb, idx