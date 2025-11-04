"""
Pipeline de experimentos K-Means para compressão de imagens.
Apenas processamento - sem visualizações.
"""

import os
import time
import gc
import numpy as np
import matplotlib.pyplot as plt

from src.clustering.kmeans import kmeans_init_centroids, run_kmeans


# ========== UTILITÁRIOS ==========

def clear_gpu_memory():
    """Limpa a memória da GPU."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        gc.collect()
    except ImportError:
        pass


def convert_to_float(img):
    """
    Converte imagem para float32 em [0,1].
    
    Returns:
        img_float, max_value, dtype
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0, img.dtype
    return img.astype(np.float32), 1.0, img.dtype


def get_optimal_index_dtype(K):
    """Retorna dtype ótimo para índices."""
    if K <= 256:
        return np.uint8
    elif K <= 65536:
        return np.uint16
    else:
        return np.uint32


def compute_compressed_size(centroids, idx, K):
    """Calcula tamanho da compressão."""
    idx_dtype = get_optimal_index_dtype(K)
    centroids_bytes = K * 3 * 4
    idx_bytes = idx.size * np.dtype(idx_dtype).itemsize
    total_bytes = centroids_bytes + idx_bytes
    return total_bytes, idx_dtype


def reconstruct_image(centroids, idx, original_shape, original_dtype):
    """Reconstrói imagem a partir de centróides e índices."""
    img_recovered = centroids[idx, :].reshape(original_shape)
    
    if original_dtype == np.uint8:
        img_recovered = np.clip(
            np.rint(img_recovered * 255.0), 0, 255
        ).astype(np.uint8)
    else:
        img_recovered = np.clip(img_recovered, 0.0, 1.0).astype(original_dtype)
    
    return img_recovered


def compute_quality_metrics(X_float, centroids, idx, max_value):
    """Calcula SSE, MSE e PSNR."""
    diffs = X_float - centroids[idx]
    sse = float(np.sum(diffs ** 2))
    
    n_channels = X_float.shape[1]
    mse = sse / (X_float.shape[0] * n_channels)
    
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 20.0 * np.log10(max_value) - 10.0 * np.log10(mse)
    
    return sse, mse, psnr


def count_unique_colors(img):
    """Conta cores únicas."""
    if img.dtype != np.uint8:
        img = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    
    colors = img.reshape(-1, 3)
    unique_colors = np.unique(colors, axis=0)
    return unique_colors.shape[0]


# ========== K-MEANS SINGLE ==========

def run_kmeans_single(X, K, max_iters=10, seed=0, n_init=1,
                      use_gpu=True, batch_size=200000):
    """
    Executa K-Means com múltiplas inicializações.
    Retorna melhor resultado por SSE.
    """
    best_result = None
    
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
        
        diffs = X - centroids[idx]
        sse = float(np.sum(diffs ** 2))
        
        if best_result is None or sse < best_result['sse']:
            best_result = {
                'centroids': centroids,
                'idx': idx,
                'sse': sse
            }
    
    return best_result['centroids'], best_result['idx'], best_result['sse']


# ========== K-MEANS GRID ==========

def run_kmeans_grid(original_img, K_list, max_iters=10, seed=0, n_init=1,
                    use_gpu=True, batch_size=200000, save_dir=None):
    """
    Executa K-Means para múltiplos valores de K.
    APENAS PROCESSAMENTO - sem visualizações.
    
    Args:
        original_img: Imagem original RGB
        K_list: Lista de valores de K
        max_iters: Iterações máximas
        seed: Seed para reprodutibilidade
        n_init: Número de inicializações por K
        use_gpu: Usar GPU se disponível
        batch_size: Tamanho do batch
        save_dir: Diretório para salvar imagens comprimidas
        
    Returns:
        results: Lista de dicts com todos os dados de cada K
    """
    # Validar
    H, W, C = original_img.shape
    assert C == 3, "Esperada imagem RGB (H, W, 3)"
    
    # Verificar GPU
    device_info = "CPU"
    if use_gpu:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            gpu_memory_gb = device.mem_info[1] / 1e9
            device_info = f"GPU ({gpu_memory_gb:.2f} GB)"
        except (ImportError, Exception):
            use_gpu = False
    
    # Preparar dados
    img_float, max_value, original_dtype = convert_to_float(original_img)
    X = img_float.reshape(-1, C)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Header
    print(f"\n{'='*70}")
    print(f"PROCESSANDO K-MEANS")
    print(f"{'='*70}")
    print(f"Resolução:   {H}×{W} ({H*W:,} pixels)")
    print(f"Device:      {device_info}")
    print(f"K valores:   {K_list}")
    print(f"Batch size:  {batch_size:,}")
    print(f"{'='*70}\n")
    
    results = []
    
    # Loop principal
    for K in K_list:
        print(f"Processando K={K}... ", end='', flush=True)
        
        if use_gpu:
            clear_gpu_memory()
        
        t0 = time.time()
        
        try:
            centroids, idx, sse = run_kmeans_single(
                X, K,
                max_iters=max_iters,
                seed=seed,
                n_init=n_init,
                use_gpu=use_gpu,
                batch_size=batch_size
            )
        except Exception as e:
            if use_gpu and 'memory' in str(e).lower():
                print("GPU OOM! Tentando CPU... ", end='', flush=True)
                clear_gpu_memory()
                centroids, idx, sse = run_kmeans_single(
                    X, K,
                    max_iters=max_iters,
                    seed=seed,
                    n_init=n_init,
                    use_gpu=False,
                    batch_size=batch_size
                )
            else:
                raise
        
        elapsed = time.time() - t0
        centroids = centroids.astype(np.float32)
        
        # Métricas
        _, mse, psnr = compute_quality_metrics(X, centroids, idx, max_value)
        
        # Reconstruir
        compressed_img = reconstruct_image(
            centroids, idx, original_img.shape, original_dtype
        )
        
        # Cores únicas
        unique_original = count_unique_colors(original_img)
        unique_compressed = count_unique_colors(compressed_img)
        
        # Tamanhos
        original_mb = (H * W * 3) / (1024 * 1024)
        compressed_bytes, idx_dtype = compute_compressed_size(centroids, idx, K)
        compressed_mb = compressed_bytes / (1024 * 1024)
        compression_ratio = original_mb / compressed_mb if compressed_mb > 0 else np.inf
        
        # Salvar imagem comprimida
        output_path = None
        if save_dir:
            output_path = os.path.join(save_dir, f"compressed_k{K}.png")
            plt.imsave(output_path, compressed_img)
        
        # Armazenar TUDO
        results.append({
            # Metadados
            'K': int(K),
            'use_gpu': bool(use_gpu),
            'max_iters': int(max_iters),
            'n_init': int(n_init),
            'tempo_s': round(float(elapsed), 4),
            
            # Dados do K-Means
            'centroids': centroids,
            'idx': idx,
            'X': X,  # Dados originais (flat)
            
            # Imagens
            'original_img': original_img,
            'compressed_img': compressed_img,
            
            # Métricas de qualidade
            'SSE': float(sse),
            'MSE': float(mse),
            'PSNR_dB': float(psnr),
            
            # Cores
            'cores_unicas_original': int(unique_original),
            'cores_unicas_comprimida': int(unique_compressed),
            
            # Tamanhos
            'tamanho_original_MB': float(original_mb),
            'tamanho_comprimido_MB': float(compressed_mb),
            'fator_compactacao': float(compression_ratio),
            'idx_dtype': str(idx_dtype.__name__),
            
            # Arquivo
            'arquivo_saida': output_path
        })
        
        print(f"✅ {elapsed:.1f}s | PSNR: {psnr:.1f} dB | "
              f"{compression_ratio:.2f}x")
        
        if use_gpu:
            clear_gpu_memory()
        gc.collect()
    
    print(f"\n{'='*70}")
    print(f"PROCESSAMENTO CONCLUÍDO! {len(results)} resultados salvos.")
    print(f"{'='*70}\n")
    
    return results