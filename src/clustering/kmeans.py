import numpy as np
from matplotlib import colors as mcolors

# Tentar importar CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_array_module(use_gpu=True):
    """Retorna numpy ou cupy baseado em disponibilidade"""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


# ========== FUNÇÕES DE DISTÂNCIA COM SUPORTE GPU ==========

def euclidean_distance(X, centroids, xp=np):
    """Distância Euclidiana (GPU-ready)"""
    # X: (m, n), centroids: (K, n)
    # Retorna: (m, K)
    return xp.linalg.norm(X[:, xp.newaxis] - centroids, axis=2)


def manhattan_distance(X, centroids, xp=np):
    """Distância Manhattan (GPU-ready)"""
    return xp.sum(xp.abs(X[:, xp.newaxis] - centroids), axis=2)


def cosine_distance(X, centroids, xp=np):
    """Distância baseada em similaridade de cosseno (GPU-ready)"""
    X_norm = X / (xp.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    C_norm = centroids / (xp.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    similarity = xp.dot(X_norm, C_norm.T)
    return 1 - similarity


def chebyshev_distance(X, centroids, xp=np):
    """Distância Chebyshev (GPU-ready)"""
    return xp.max(xp.abs(X[:, xp.newaxis] - centroids), axis=2)


def minkowski_distance(X, centroids, p=3, xp=np):
    """Distância Minkowski (GPU-ready)"""
    return xp.sum(xp.abs(X[:, xp.newaxis] - centroids) ** p, axis=2) ** (1/p)


DISTANCE_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'chebyshev': chebyshev_distance,
    'minkowski': minkowski_distance,
}


# ========== CONVERSÕES DE ESPAÇO DE COR ==========

def rgb_to_hsv_vectorized(rgb_array):
    """Converte RGB para HSV (mantém na CPU por limitação do matplotlib)"""
    # Matplotlib não suporta GPU, então converter temporariamente
    is_gpu = False
    if CUPY_AVAILABLE and isinstance(rgb_array, cp.ndarray):
        rgb_array = cp.asnumpy(rgb_array)
        is_gpu = True
    
    if rgb_array.ndim == 2:
        h, w = 1, rgb_array.shape[0]
        rgb_img = rgb_array.reshape(h, w, 3)
        hsv_img = mcolors.rgb_to_hsv(rgb_img)
        result = hsv_img.reshape(-1, 3)
    else:
        result = mcolors.rgb_to_hsv(rgb_array)
    
    if is_gpu:
        result = cp.asarray(result)
    
    return result


def hsv_to_rgb_vectorized(hsv_array):
    """Converte HSV para RGB (mantém na CPU por limitação do matplotlib)"""
    is_gpu = False
    if CUPY_AVAILABLE and isinstance(hsv_array, cp.ndarray):
        hsv_array = cp.asnumpy(hsv_array)
        is_gpu = True
    
    if hsv_array.ndim == 2:
        h, w = 1, hsv_array.shape[0]
        hsv_img = hsv_array.reshape(h, w, 3)
        rgb_img = mcolors.hsv_to_rgb(hsv_img)
        result = rgb_img.reshape(-1, 3)
    else:
        result = mcolors.hsv_to_rgb(hsv_array)
    
    if is_gpu:
        result = cp.asarray(result)
    
    return result


# ========== K-MEANS COM GPU ==========

def find_closest_centroids(X, centroids, distance_metric='euclidean', use_gpu=True):
    """
    Encontra o centróide mais próximo (GPU-ready).
    """
    xp = get_array_module(use_gpu)
    
    if isinstance(distance_metric, str):
        if distance_metric not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Métrica '{distance_metric}' não reconhecida.")
        distance_func = DISTANCE_FUNCTIONS[distance_metric]
    elif callable(distance_metric):
        distance_func = distance_metric
    else:
        raise TypeError("distance_metric deve ser string ou função")
    
    # Calcular distâncias
    distances = distance_func(X, centroids, xp=xp)
    
    # Encontrar índice do mínimo
    idx = xp.argmin(distances, axis=1).astype(int)
    
    return idx


def compute_centroids(X, idx, K, use_gpu=True):
    """Calcula os novos centróides (GPU-ready)."""
    xp = get_array_module(use_gpu)
    
    m, n = X.shape
    centroids = xp.zeros((K, n), dtype=X.dtype)

    for i in range(K):
        mask = (idx == i)
        points_assigned = X[mask]
        if xp.sum(mask) > 0:
            centroids[i] = xp.mean(points_assigned, axis=0)
    
    return centroids


def kMeans_init_centroids(X, K, use_gpu=True):
    """Inicializa os centróides (GPU-ready)."""
    xp = get_array_module(use_gpu)
    
    if CUPY_AVAILABLE and isinstance(X, cp.ndarray):
        # Usar random do CuPy
        randidx = cp.random.permutation(X.shape[0])
    else:
        randidx = np.random.permutation(X.shape[0])
    
    centroids = X[randidx[:K]]
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False, 
               plot_function=None, distance_metric='euclidean', 
               color_space='rgb', use_gpu=True):
    """
    Executa o algoritmo K-Means (GPU-ready).
    
    Parâmetros:
        use_gpu: bool - Se True, tenta usar GPU. Se False, usa CPU.
    """
    xp = get_array_module(use_gpu)
    
    # Mensagem sobre GPU
    if use_gpu and CUPY_AVAILABLE:
        print(f"🚀 Executando K-Means na GPU")
    else:
        if use_gpu and not CUPY_AVAILABLE:
            print(f"⚠️  GPU solicitada mas CuPy não disponível, usando CPU")
        else:
            print(f"💻 Executando K-Means na CPU")
    
    # Converter X para GPU se necessário
    if use_gpu and CUPY_AVAILABLE and not isinstance(X, cp.ndarray):
        X = cp.asarray(X)
    elif not use_gpu and CUPY_AVAILABLE and isinstance(X, cp.ndarray):
        X = cp.asnumpy(X)
    
    # Converter para espaço de cor
    if color_space == 'hsv':
        print(f"🎨 Convertendo RGB → HSV")
        X_transformed = rgb_to_hsv_vectorized(X)
        initial_centroids_transformed = rgb_to_hsv_vectorized(initial_centroids)
    elif color_space == 'hls':
        print(f"🎨 Convertendo RGB → HLS")
        # HLS não implementado para GPU ainda
        if use_gpu and CUPY_AVAILABLE:
            X = cp.asnumpy(X)
        X_transformed = X  # Placeholder
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
            # Para plot, precisa estar na CPU
            if CUPY_AVAILABLE and isinstance(X_transformed, cp.ndarray):
                X_plot = cp.asnumpy(X_transformed)
                centroids_plot = cp.asnumpy(centroids)
                prev_plot = cp.asnumpy(previous_centroids)
                idx_plot = cp.asnumpy(idx)
            else:
                X_plot = X_transformed
                centroids_plot = centroids
                prev_plot = previous_centroids
                idx_plot = idx
            
            plot_function(X_plot, centroids_plot, prev_plot, idx_plot, K, i)
            previous_centroids = centroids.copy()
            
        centroids = compute_centroids(X_transformed, idx, K, use_gpu=use_gpu)
    
    # Converter de volta para RGB se necessário
    if color_space == 'hsv':
        centroids_rgb = hsv_to_rgb_vectorized(centroids)
    else:
        centroids_rgb = centroids
    
    # Retornar sempre na CPU para compatibilidade
    if CUPY_AVAILABLE and isinstance(centroids_rgb, cp.ndarray):
        centroids_rgb = cp.asnumpy(centroids_rgb)
    if CUPY_AVAILABLE and isinstance(idx, cp.ndarray):
        idx = cp.asnumpy(idx)
    
    return centroids_rgb, idx