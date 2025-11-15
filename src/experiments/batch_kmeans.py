"""Pipeline K-Means para compressão de imagens - processamento simples."""

import gc
import time
import numpy as np

from src.clustering.kmeans import kmeans_init_centroids, run_kmeans


def clear_gpu_memory():
    """Limpa memória da GPU."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    except ImportError:
        pass


def get_optimal_dtype(K):
    """Retorna dtype ótimo para índices."""
    if K <= 256:
        return np.uint8
    elif K <= 65536:
        return np.uint16
    return np.uint32


def compute_quality_metrics(X_float, centroids, idx):
    """
    Calcula SSE, MSE e PSNR.
    
    X_float e centroids normalizados [0,1], então max_value=1.0 para PSNR.
    """
    diffs = X_float - centroids[idx]
    sse = float(np.sum(diffs ** 2))
    
    mse = sse / (X_float.shape[0] * X_float.shape[1])
    
    if mse < 1e-10:
        return sse, mse, float('inf')
    
    psnr = -10.0 * np.log10(mse)
    return sse, mse, psnr


def reconstruct_image(centroids, idx, shape, dtype):
    """Reconstrói imagem a partir de centróides e índices."""
    img = centroids[idx, :].reshape(shape)
    
    if dtype == np.uint8:
        return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    
    return np.clip(img, 0.0, 1.0).astype(dtype)


def count_unique_colors(img):
    """Conta cores únicas."""
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    
    colors = img.reshape(-1, 3)
    return np.unique(colors, axis=0).shape[0]


def run_kmeans_single(X, K, max_iters=10, seed=0, n_init=1,
                      use_gpu=True, batch_size=200000):
    """
    Executa K-Means com múltiplas inicializações.
    
    Retorna melhor resultado por SSE.
    """
    best_sse = float('inf')
    best_centroids = None
    best_idx = None
    
    for i in range(n_init):
        if seed is not None:
            np.random.seed(seed + i)
        
        initial_centroids = kmeans_init_centroids(X, K, use_gpu=use_gpu)
        centroids, idx = run_kmeans(
            X, initial_centroids,
            max_iters=max_iters,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        
        sse = float(np.sum((X - centroids[idx]) ** 2))
        
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            best_idx = idx
    
    return best_centroids, best_idx, best_sse


def compress_image(img, K, max_iters=10, seed=0, n_init=1,
                   use_gpu=True, batch_size=200000):
    """
    Comprime imagem usando K-Means.
    
    Args:
        img: Imagem RGB (H, W, 3)
        K: Número de cores/clusters
        max_iters: Iterações máximas
        seed: Seed para reprodutibilidade
        n_init: Número de inicializações
        use_gpu: Usar GPU se disponível
        batch_size: Tamanho do batch
        
    Returns:
        dict com resultados completos
    """
    H, W, C = img.shape
    assert C == 3, "Esperada imagem RGB (H, W, 3)"
    
    # Preparar dados
    original_dtype = img.dtype
    img_float = img.astype(np.float32) / (255.0 if img.dtype == np.uint8
                                           else 1.0)
    X = img_float.reshape(-1, C)
    
    if use_gpu:
        clear_gpu_memory()
    
    print(f"Comprimindo com K={K}... ", end='', flush=True)
    t0 = time.time()
    
    try:
        centroids, idx, sse = run_kmeans_single(
            X, K, max_iters, seed, n_init, use_gpu, batch_size
        )
    except Exception as e:
        if use_gpu and 'memory' in str(e).lower():
            print("GPU OOM! CPU... ", end='', flush=True)
            clear_gpu_memory()
            centroids, idx, sse = run_kmeans_single(
                X, K, max_iters, seed, n_init, False, batch_size
            )
        else:
            raise
    
    elapsed = time.time() - t0
    centroids = centroids.astype(np.float32)
    
    # Métricas
    _, mse, psnr = compute_quality_metrics(X, centroids, idx)
    
    # Reconstruir
    compressed_img = reconstruct_image(centroids, idx, img.shape,
                                       original_dtype)
    
    # Cores
    unique_orig = count_unique_colors(img)
    unique_comp = count_unique_colors(compressed_img)
    
    # Tamanhos
    orig_mb = (H * W * 3) / (1024 * 1024)
    idx_dtype = get_optimal_dtype(K)
    comp_bytes = K * 3 * 4 + idx.size * np.dtype(idx_dtype).itemsize
    comp_mb = comp_bytes / (1024 * 1024)
    ratio = orig_mb / comp_mb if comp_mb > 0 else np.inf
    
    print(f"✅ {elapsed:.1f}s | PSNR: {psnr:.1f} dB | {ratio:.2f}x")
    
    if use_gpu:
        clear_gpu_memory()
    gc.collect()
    
    return {
        'K': K,
        'tempo_s': round(elapsed, 4),
        'centroids': centroids,
        'idx': idx,
        'compressed_img': compressed_img,
        'SSE': sse,
        'MSE': mse,
        'PSNR_dB': psnr,
        'cores_originais': unique_orig,
        'cores_comprimidas': unique_comp,
        'tamanho_original_MB': orig_mb,
        'tamanho_comprimido_MB': comp_mb,
        'fator_compactacao': ratio,
        'idx_dtype': str(idx_dtype.__name__)
    }