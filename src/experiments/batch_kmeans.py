import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.clustering.kmeans import kMeans_init_centroids, run_kMeans, DISTANCE_FUNCTIONS
from src.visualization.plot_3d import plot_kMeans_RGB, show_centroid_colors


# -------------------------------
# Utilitários
# -------------------------------
def _img_to_float01(img):
    """Converte para float32 em [0,1] e retorna também o valor máximo original (1.0 ou 255.0) e o dtype."""
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0), 255.0, img.dtype
    return img.astype(np.float32), 1.0, img.dtype


def _reconstruct_image(centroids, idx, original_shape, original_dtype):
    """
    Reconstrói imagem a partir de centróides/índices (centroids em [0,1]) e volta ao dtype original.
    """
    X_recovered = centroids[idx, :].reshape(original_shape)
    if original_dtype == np.uint8:
        X_recovered = np.clip(np.rint(X_recovered * 255.0), 0, 255).astype(np.uint8)
    else:
        X_recovered = np.clip(X_recovered, 0.0, 1.0).astype(original_dtype)
    return X_recovered


def _compute_sse_mse_psnr(X_float01, centroids, idx, max_i):
    """
    Calcula SSE, MSE e PSNR comparando a imagem original (achatada) com a reconstruída (por idx).
    """
    diffs = X_float01 - centroids[idx]
    sse = float(np.sum(diffs * diffs))
    C = X_float01.shape[1]
    mse = sse / (X_float01.shape[0] * C)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20.0 * np.log10(max_i) - 10.0 * np.log10(mse)
    return sse, mse, psnr


def plot_comparison_with_stats(original_img, X_recovered, centroids, idx, K, 
                               distance_metric='euclidean', color_space='rgb'):
    """
    Mostra (lado a lado) original vs comprimida e inclui:
      - Tamanho original e "comprimido" (centróides + índices)
      - Nº de cores únicas em cada imagem
      - Percentual de redução
      - Métrica de distância usada
      - Espaço de cor usado
    Retorna o objeto Figure para permitir salvar se necessário.
    """
    H, W, C = original_img.shape
    assert C == 3, "Esperada imagem RGB (H, W, 3)."

    def to_uint8(arr):
        if arr.dtype == np.uint8:
            return arr
        return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)

    orig_u8 = to_uint8(original_img)
    rec_u8  = to_uint8(X_recovered)
    unique_colors_original   = np.unique(orig_u8.reshape(-1, C), axis=0).shape[0]
    unique_colors_compressed = np.unique(rec_u8.reshape(-1, C), axis=0).shape[0]

    original_size_mb   = original_img.nbytes / (1024 * 1024)
    compressed_size_mb = (centroids.nbytes + idx.nbytes) / (1024 * 1024)
    reducao_pct = (1 - compressed_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0.0

    fig, ax = plt.subplots(1, 2, figsize=(16, 16))
    plt.axis('off')

    ax[0].imshow(original_img)
    ax[0].set_title(
        f'Original\nTamanho: {original_size_mb:.2f} MB\nCores únicas: {unique_colors_original:,}',
        fontsize=14
    )
    ax[0].set_axis_off()

    ax[1].imshow(X_recovered)
    ax[1].set_title(
        f'Comprimida com {K} cores\nTamanho: {compressed_size_mb:.2f} MB\nCores únicas: {unique_colors_compressed:,}\nMétrica: {distance_metric} | Espaço: {color_space.upper()}',
        fontsize=14
    )
    ax[1].set_axis_off()

    plt.suptitle(
        f'Redução de {original_size_mb:.2f} MB para {compressed_size_mb:.2f} MB '
        f'({reducao_pct:.1f}% menor)\n'
        f'Cores reduzidas de {unique_colors_original:,} para {unique_colors_compressed:,}',
        fontsize=16, y=0.98
    )
    plt.tight_layout()
    return fig


# -------------------------------
# Núcleo do experimento
# -------------------------------
def run_kmeans_single(X_float01, K, max_iters=10, seed=0, n_init=1, 
                      distance_metric='euclidean', color_space='rgb', use_gpu=True):
    """
    Roda K-Means para um K com n_init inicializações.
    Retorna os melhores centroids/idx por SSE.
    
    Parâmetros:
        X_float01: dados em [0,1]
        K: número de clusters
        max_iters: iterações máximas
        seed: seed para reprodutibilidade
        n_init: número de inicializações
        distance_metric: métrica de distância
        color_space: 'rgb', 'hsv' ou 'hls'
        use_gpu: usar GPU se disponível
    """
    best = None
    for rep in range(n_init):
        if seed is not None:
            np.random.seed(seed + rep)
        init = kMeans_init_centroids(X_float01, K, use_gpu=use_gpu)
        centroids, idx = run_kMeans(
            X_float01, 
            init, 
            max_iters=max_iters, 
            distance_metric=distance_metric,
            color_space=color_space,
            use_gpu=use_gpu
        )
        diffs = X_float01 - centroids[idx]
        sse = float(np.sum(diffs * diffs))
        if (best is None) or (sse < best["sse"]):
            best = {"centroids": centroids, "idx": idx, "sse": sse}
    return best["centroids"], best["idx"], best["sse"]


def run_kmeans_grid(original_img, K_list, max_iters=10, seed=0, n_init=1,
                    save_dir=None, plot_each=True, plot_rgb=False, show_palette=False, 
                    save_plots=False, distance_metric='euclidean', 
                    color_space='rgb', use_gpu=True):
    """
    Executa o pipeline para vários K e retorna uma lista de métricas por K.
    
    Parâmetros:
        original_img: imagem original
        K_list: lista de valores de K para testar
        max_iters: número máximo de iterações
        seed: seed para reprodutibilidade
        n_init: número de inicializações por K
        save_dir: diretório para salvar resultados
        plot_each: plotar comparação para cada K
        plot_rgb: plotar no espaço RGB 3D
        show_palette: mostrar paleta de cores
        save_plots: salvar plots em arquivo
        distance_metric: 'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'
        color_space: 'rgb', 'hsv' ou 'hls'
        use_gpu: usar GPU se disponível
    """
    # Validações
    if isinstance(distance_metric, str) and distance_metric not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Métrica '{distance_metric}' inválida. Use: {list(DISTANCE_FUNCTIONS.keys())}")
    
    if color_space not in ['rgb', 'hsv', 'hls']:
        raise ValueError(f"color_space deve ser 'rgb', 'hsv' ou 'hls', não '{color_space}'")
    
    # Verificar GPU
    device_info = "CPU"
    if use_gpu:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            device_info = f"GPU (Memória: {device.mem_info[1] / 1e9:.2f} GB)"
        except:
            device_info = "CPU (GPU não disponível)"
    
    # Preparação
    img01, max_i, original_dtype = _img_to_float01(original_img)
    H, W, C = img01.shape
    assert C == 3, "Esperada imagem RGB de 3 canais."
    X = img01.reshape(-1, C)

    results = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXECUTANDO K-MEANS")
    print(f"{'='*70}")
    print(f"Métrica de distância: {distance_metric.upper()}")
    print(f"Espaço de cor:        {color_space.upper()}")
    print(f"Device:               {device_info}")
    print(f"Valores de K:         {K_list}")
    print(f"{'='*70}\n")

    for K in K_list:
        print(f"\n--- Processando K={K} ---")
        t0 = time.time()
        centroids, idx, sse = run_kmeans_single(
            X, K, 
            max_iters=max_iters, 
            seed=seed, 
            n_init=n_init, 
            distance_metric=distance_metric,
            color_space=color_space,
            use_gpu=use_gpu
        )
        elapsed = time.time() - t0

        # Métricas principais
        _, mse, psnr = _compute_sse_mse_psnr(X, centroids, idx, max_i)

        # Reconstrução
        rec_img = _reconstruct_image(centroids, idx, original_img.shape, original_dtype)

        # Cores únicas e tamanhos
        def to_uint8(arr):
            if arr.dtype == np.uint8:
                return arr
            return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)

        orig_u8 = to_uint8(original_img)
        rec_u8  = to_uint8(rec_img)
        unique_colors_original   = np.unique(orig_u8.reshape(-1, C), axis=0).shape[0]
        unique_colors_compressed = np.unique(rec_u8.reshape(-1, C), axis=0).shape[0]
        original_mb   = original_img.nbytes / (1024 * 1024)
        compressed_mb = (centroids.nbytes + idx.nbytes) / (1024 * 1024)
        ratio = (original_mb / compressed_mb) if compressed_mb > 0 else np.inf

        # Plot ao final de cada K
        if plot_each:
            fig = plot_comparison_with_stats(original_img, rec_img, centroids, idx, K, 
                                            distance_metric, color_space)
            plt.show()
            if save_dir and save_plots:
                fig_path = os.path.join(save_dir, f"plot_k{K}_{distance_metric}_{color_space}.png")
                fig.savefig(fig_path, bbox_inches="tight")
                plt.close(fig)

        # Plots extras
        if plot_rgb:
            plot_kMeans_RGB(X, centroids, idx, K)
        if show_palette:
            show_centroid_colors(centroids)

        # Salvar imagem reconstruída
        out_path = None
        if save_dir:
            out_path = os.path.join(save_dir, f"image_k{K}_{distance_metric}_{color_space}.png")
            plt.imsave(out_path, rec_img)

        # Registrar resultados
        results.append({
            "K": int(K),
            "distance_metric": str(distance_metric),
            "color_space": str(color_space),
            "use_gpu": bool(use_gpu),
            "max_iters": int(max_iters),
            "n_init": int(n_init),
            "tempo_s": round(float(elapsed), 4),
            "SSE": float(sse),
            "MSE": float(mse),
            "PSNR_dB": float(psnr),
            "cores_unicas_original": int(unique_colors_original),
            "cores_unicas_comprimida": int(unique_colors_compressed),
            "tamanho_original_MB": float(original_mb),
            "tamanho_comprimido_MB": float(compressed_mb),
            "fator_compactacao": float(ratio),
            "arquivo_saida": out_path
        })
        
        # Mostrar resumo
        print(f"  ✅ Concluído em {elapsed:.2f}s")
        print(f"     PSNR: {psnr:.2f} dB")
        print(f"     Cores: {unique_colors_original:,} → {unique_colors_compressed:,}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO CONCLUÍDO!")
    print(f"{'='*70}\n")

    return results