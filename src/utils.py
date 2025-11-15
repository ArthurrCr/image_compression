import gc
import time
import numpy as np
import os
import imageio.v2 as imageio
import io
from PIL import Image
from .kmeans import kmeans_init_centroids, run_kmeans, clear_gpu_memory


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


def make_palette_image(result):
    """
    Constrói uma imagem indexada (modo 'P') a partir do resultado do compress_image.
    
    Usa:
        - result['centroids']: paleta Kx3 em [0,1]
        - result['idx']: índices (H*W,)
        - result['compressed_img']: para recuperar H e W
    
    Retorna:
        imgP: PIL.Image em modo 'P' (paleta), pronta para salvar como PNG-8.
    """
    K = result['K']
    if K > 256:
        raise ValueError(
            f"PNG com paleta suporta no máximo 256 cores, mas K={K}."
        )
    
    centroids = result['centroids']       # (K, 3) em [0,1]
    idx = result['idx']                   # (H*W,)
    comp = result['compressed_img']       # (H, W, 3) – só para shape
    H, W = comp.shape[:2]
    
    # Índices em formato 2D e 8 bits por pixel
    idx_2d = idx.reshape(H, W).astype(np.uint8)
    
    # Paleta em uint8 [0,255]
    palette = np.clip(np.rint(centroids * 255.0), 0, 255).astype(np.uint8)
    
    # PIL espera lista flat com exatamente 256*3 valores (R,G,B de cada cor)
    pal_list = palette.reshape(-1).tolist()
    pal_list += [0] * (256 * 3 - len(pal_list))  # completa até 256 cores
    
    # Cria imagem em modo 'P' (paleta) usando os índices
    imgP = Image.fromarray(idx_2d, mode="P")
    imgP.putpalette(pal_list)
    return imgP


def compute_png_size_rgb_mb(img, optimize=False):
    """
    Codifica uma imagem RGB como PNG em memória e retorna o tamanho em MB.

    Usa sempre PNG truecolor (24 bits). Não salva em disco.

    Args:
        img: numpy array (H, W, 3), idealmente uint8.
        optimize: se True, ativa optimize=True no encoder PNG
                  (use o MESMO valor para original e quantizada se quiser
                  uma comparação justa).
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    pil_img = Image.fromarray(img_u8, mode="RGB")
    buf = io.BytesIO()
    if optimize:
        pil_img.save(buf, format="PNG", optimize=True)
    else:
        pil_img.save(buf, format="PNG")
    size_bytes = len(buf.getvalue())
    buf.close()
    return size_bytes / (1024 * 1024)


def compute_png_size_palette_mb(result, optimize=False):
    """
    Codifica em memória o PNG-8 (paleta) gerado a partir do resultado do K-Means
    e retorna o tamanho em MB.

    Args:
        result: dict retornado por compress_image(...)
        optimize: mesmo significado da função RGB.
    """
    imgP = make_palette_image(result)
    buf = io.BytesIO()
    if optimize:
        imgP.save(buf, format="PNG", optimize=True)
    else:
        imgP.save(buf, format="PNG")
    size_bytes = len(buf.getvalue())
    buf.close()
    return size_bytes / (1024 * 1024)



def compress_image(img, K, max_iters=10, seed=0, n_init=1,
                   use_gpu=True, batch_size=200000):
    """
    Comprime imagem usando K-Means.

    Obs:
      - 'tamanho_original_MB' e 'tamanho_comprimido_MB' aqui são da
        REPRESENTAÇÃO EM MEMÓRIA (RGB cru vs paleta+índice).
      - Tamanhos REAIS de arquivo (PNG) você obtém com
        compute_png_size_rgb_mb / compute_png_size_palette_mb.
    
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
    
    # Reconstruir (imagem RGB para visualização)
    compressed_img = reconstruct_image(centroids, idx, img.shape,
                                       original_dtype)
    
    # Cores
    unique_orig = count_unique_colors(img)
    unique_comp = count_unique_colors(compressed_img)
    
    # Tamanhos TEÓRICOS (representação crua RGB vs índice+paleta)
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
        'tamanho_original_MB': orig_mb,       # memória RGB crua
        'tamanho_comprimido_MB': comp_mb,     # memória paleta+índice
        'fator_compactacao': ratio,
        'idx_dtype': str(idx_dtype.__name__),
    }



def run_full_experiment(image_path, K, max_iters=10, seed=0,
                        n_init=1, use_gpu=True, batch_size=200000):
    """
    Carrega a imagem do disco, roda o K-Means e monta o dict de resultado
    já com:
      - original_img
      - jpeg_size_mb (tamanho real do arquivo original)
    
    Args:
        image_path: caminho do arquivo de imagem (ex: JPEG da câmera)
        K, max_iters, seed, n_init, use_gpu, batch_size: mesmos do compress_image

    Returns:
        result: dict compatível com plot_result / plot_comparison
    """
    # 1) Carrega a imagem (como array RGB)
    original_img = imageio.imread(image_path)

    # 2) Roda o K-Means usando sua compress_image
    result = compress_image(
        original_img,
        K=K,
        max_iters=max_iters,
        seed=seed,
        n_init=n_init,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    # 3) Tamanho real do arquivo original em disco
    file_size_bytes = os.path.getsize(image_path)
    jpeg_size_mb = file_size_bytes / (1024 * 1024)

    # 4) Anexa informações extras pro pipeline de visualização
    result["original_img"] = original_img
    result["jpeg_size_mb"] = jpeg_size_mb

    return result