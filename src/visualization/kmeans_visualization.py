"""
Funções de visualização para K-Means em compressão de imagens.
Inclui plots 3D, paletas de cores, análises de compressão e zoom.
Consolidação de plot_3d.py e visualization.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ========== VISUALIZAÇÕES 3D E PALETA ==========

def plot_kmeans_rgb(X, centroids, idx, K, max_points=10000):
    """
    Plota resultado do K-Means no espaço RGB 3D (otimizado com subsampling).
    
    Args:
        X: Dados (n_samples, 3)
        centroids: Centróides (K, 3)
        idx: Índices dos clusters (n_samples,)
        K: Número de clusters
        max_points: Número máximo de pontos a plotar (default: 10000)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalizar para [0,1]
    X_plot = X / 255.0 if X.max() > 1.0 else X
    centroids_plot = centroids / 255.0 if centroids.max() > 1.0 else centroids
    
    n_samples = X_plot.shape[0]
    
    # Subsampling se necessário
    if n_samples > max_points:
        sample_indices = np.random.choice(n_samples, max_points, replace=False)
        X_sampled = X_plot[sample_indices]
        idx_sampled = idx[sample_indices]
        print(f"   ⚡ Subsampling: {n_samples:,} → {max_points:,} pontos")
    else:
        X_sampled = X_plot
        idx_sampled = idx
    
    # Plotar todos os pontos de uma vez (muito mais rápido)
    colors = centroids_plot[idx_sampled]
    ax.scatter(
        X_sampled[:, 0],
        X_sampled[:, 1],
        X_sampled[:, 2],
        c=colors,
        s=3,
        alpha=0.5
    )
    
    # Plotar centróides com destaque
    ax.scatter(
        centroids_plot[:, 0],
        centroids_plot[:, 1],
        centroids_plot[:, 2],
        c=centroids_plot,
        s=200,
        marker='*',
        edgecolors='black',
        linewidths=2,
        alpha=1.0,
        label='Centróides'
    )
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(f'K-Means no espaço RGB (K={K})', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


def show_centroid_colors(centroids):
    """
    Mostra paleta de cores dos centróides.
    
    Args:
        centroids: Centróides (K, 3)
    """
    K = centroids.shape[0]
    
    # Normalizar para [0,1]
    colors = centroids / 255.0 if centroids.max() > 1.0 else centroids
    
    fig, axes = plt.subplots(1, K, figsize=(K * 2, 2))
    if K == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.imshow([[colors[i]]])
        ax.axis('off')
        ax.set_title(f'Cor {i}', fontsize=10)
    
    plt.suptitle("Paleta de Cores", fontsize=12, y=0.85)
    plt.tight_layout()
    plt.show()


# ========== ANÁLISE DE COMPRESSÃO ==========

def get_index_dtype_info(K):
    """
    Retorna informações sobre dtype de índices.
    
    Args:
        K: Número de clusters
        
    Returns:
        dtype, bytes_per_pixel, range_str
    """
    if K <= 256:
        return np.uint8, 1, "0 a 255"
    elif K <= 65536:
        return np.uint16, 2, "0 a 65,535"
    else:
        return np.uint32, 4, "0 a 4,294,967,295"


# ========== ZOOM COMPARATIVO ==========

def plot_zoom_comparison(original_img, compressed_img, K, 
                        zoom_size=200, seed=None):
    """
    Plota zoom em região aleatória comparando original vs comprimida.
    
    Args:
        original_img: Imagem original
        compressed_img: Imagem comprimida
        K: Número de cores
        zoom_size: Tamanho da região de zoom (pixels)
        seed: Seed para posição aleatória
    """
    H, W, C = original_img.shape
    
    # Ajustar tamanho do zoom
    zoom_h = min(zoom_size, H)
    zoom_w = min(zoom_size, W)
    
    if zoom_h <= 0 or zoom_w <= 0:
        print("⚠️  Imagem muito pequena para zoom")
        return
    
    # Escolher posição aleatória
    if seed is not None:
        np.random.seed(seed)
    
    max_y = H - zoom_h
    max_x = W - zoom_w
    
    if max_y <= 0 or max_x <= 0:
        print("⚠️  Região de zoom não cabe na imagem")
        return
    
    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)
    end_y = start_y + zoom_h
    end_x = start_x + zoom_w
    
    # Extrair regiões
    zoom_orig = original_img[start_y:end_y, start_x:end_x]
    zoom_comp = compressed_img[start_y:end_y, start_x:end_x]
    
    # Calcular métricas do zoom
    def to_float(img):
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
    zoom_orig_f = to_float(zoom_orig)
    zoom_comp_f = to_float(zoom_comp)
    
    mse_zoom = np.mean((zoom_orig_f - zoom_comp_f) ** 2)
    psnr_zoom = (20 * np.log10(1.0) - 10 * np.log10(mse_zoom) 
                 if mse_zoom > 0 else float('inf'))
    
    colors_orig = len(np.unique(zoom_orig.reshape(-1, 3), axis=0))
    colors_comp = len(np.unique(zoom_comp.reshape(-1, 3), axis=0))
    
    # Plotar
    fig = plt.figure(figsize=(20, 10))
    
    # Imagem completa original
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original - Imagem Completa', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Retângulo indicando zoom
    rect = Rectangle((start_x, start_y), zoom_w, zoom_h,
                     linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    
    # Imagem completa comprimida
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(compressed_img)
    ax2.set_title(f'Comprimida (K={K}) - Imagem Completa',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    rect2 = Rectangle((start_x, start_y), zoom_w, zoom_h,
                      linewidth=3, edgecolor='red', facecolor='none')
    ax2.add_patch(rect2)
    
    # Zoom original
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(zoom_orig)
    ax3.set_title('Zoom Original', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Grid para pixels individuais
    if zoom_size <= 50:
        ax3.set_xticks(np.arange(-0.5, zoom_w, 1), minor=True)
        ax3.set_yticks(np.arange(-0.5, zoom_h, 1), minor=True)
        ax3.grid(which='minor', color='gray', linestyle='-',
                linewidth=0.5, alpha=0.3)
    
    # Zoom comprimido
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(zoom_comp)
    ax4.set_title('Zoom Comprimida', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    if zoom_size <= 50:
        ax4.set_xticks(np.arange(-0.5, zoom_w, 1), minor=True)
        ax4.set_yticks(np.arange(-0.5, zoom_h, 1), minor=True)
        ax4.grid(which='minor', color='gray', linestyle='-',
                linewidth=0.5, alpha=0.3)
    
    # Título geral
    reduction_pct = (1 - colors_comp / colors_orig) * 100 if colors_orig > 0 else 0
    plt.suptitle(
        f'Comparação com Zoom - K={K}\n'
        f'Região: PSNR={psnr_zoom:.2f} dB | '
        f'Cores: {colors_orig} → {colors_comp} '
        f'({reduction_pct:.1f}% redução)',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    plt.show()
    
    # Informações resumidas
    print(f"\n🔍 ZOOM:")
    print(f"   Posição: ({start_x}, {start_y}) - "
          f"Tamanho: {zoom_w}×{zoom_h}px")
    print(f"   PSNR: {psnr_zoom:.2f} dB")
    print(f"   Cores: {colors_orig} → {colors_comp} "
          f"({reduction_pct:.1f}% redução)\n")


# ========== PLOT INDIVIDUAL ==========

def plot_single_result(result, show_comparison=True, show_rgb=False,
                       show_palette=False, show_analysis=False,
                       show_zoom=False, zoom_size=200, max_points=10000):
    """
    Plota visualizações para um único resultado.
    
    Args:
        result: Dict de resultado do run_kmeans_grid
        show_comparison: Mostrar comparação lado-a-lado
        show_rgb: Mostrar plot 3D RGB
        show_palette: Mostrar paleta de cores
        show_analysis: Mostrar análise de compressão
        show_zoom: Mostrar zoom comparativo
        zoom_size: Tamanho da região de zoom
        max_points: Número máximo de pontos para plot RGB 3D
    """
    K = result['K']
    print(f"\n{'='*70}")
    print(f"VISUALIZANDO K={K}")
    print(f"{'='*70}\n")
    
    # 1. Comparação lado-a-lado
    if show_comparison:
        plot_comparison_sidebyside(result)
    
    # 2. Plot 3D RGB
    if show_rgb:
        plot_kmeans_rgb(
            result['X'],
            result['centroids'],
            result['idx'],
            K,
            max_points=max_points
        )
    
    # 3. Paleta de cores
    if show_palette:
        show_centroid_colors(result['centroids'])
    
    # 4. Zoom comparativo
    if show_zoom:
        plot_zoom_comparison(
            result['original_img'],
            result['compressed_img'],
            K,
            zoom_size=zoom_size,
            seed=42
        )


def plot_comparison_sidebyside(result):
    """
    Plota comparação lado-a-lado para um resultado.
    
    Args:
        result: Dict de resultado
    """
    original_img = result['original_img']
    compressed_img = result['compressed_img']
    K = result['K']
    
    # Estatísticas
    unique_original = result['cores_unicas_original']
    unique_compressed = result['cores_unicas_comprimida']
    original_mb = result['tamanho_original_MB']
    compressed_mb = result['tamanho_comprimido_MB']
    ratio = result['fator_compactacao']
    psnr = result['PSNR_dB']
    
    reduction_pct = (1 - compressed_mb / original_mb) * 100
    
    # Plotar
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(original_img)
    ax[0].set_title(
        f'Original\n'
        f'{original_mb:.2f} MB | {unique_original:,} cores',
        fontsize=14
    )
    ax[0].axis('off')
    
    ax[1].imshow(compressed_img)
    ax[1].set_title(
        f'Comprimida (K={K})\n'
        f'{compressed_mb:.2f} MB | {unique_compressed:,} cores',
        fontsize=14
    )
    ax[1].axis('off')
    
    plt.suptitle(
        f'Compressão {ratio:.2f}x ({reduction_pct:.1f}% redução) | '
        f'PSNR: {psnr:.1f} dB',
        fontsize=16,
        y=0.95
    )
    
    plt.tight_layout()
    plt.show()


# ========== PLOT MÚLTIPLOS ==========

def plot_all_comparisons(results):
    """
    Plota comparações lado-a-lado para todos os resultados.
    
    Args:
        results: Lista de resultados do run_kmeans_grid
    """
    for result in results:
        plot_comparison_sidebyside(result)


def plot_all_rgb(results, max_points=10000):
    """
    Plota visualizações RGB 3D para todos os resultados em subplots (otimizado).
    
    Args:
        results: Lista de resultados
        max_points: Número máximo de pontos por plot
    """
    n_results = len(results)
    
    if n_results == 0:
        print("⚠️  Nenhum resultado para plotar")
        return
    
    # Calcular layout de subplots
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(8 * n_cols, 6 * n_rows))
    
    print(f"\n{'='*60}")
    print(f"PLOTANDO {n_results} GRÁFICOS RGB 3D")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        K = result['K']
        X = result['X']
        centroids = result['centroids']
        idx = result['idx']
        
        print(f"\n[{i}/{n_results}] Processando K={K}...")
        
        ax = fig.add_subplot(n_rows, n_cols, i, projection='3d')
        
        # Normalizar para [0,1]
        X_plot = X / 255.0 if X.max() > 1.0 else X
        centroids_plot = centroids / 255.0 if centroids.max() > 1.0 else centroids
        
        n_samples = X_plot.shape[0]
        
        # Subsampling se necessário
        if n_samples > max_points:
            sample_indices = np.random.choice(n_samples, max_points, replace=False)
            X_sampled = X_plot[sample_indices]
            idx_sampled = idx[sample_indices]
            print(f"   ⚡ Subsampling: {n_samples:,} → {max_points:,} pontos")
        else:
            X_sampled = X_plot
            idx_sampled = idx
        
        # Plotar todos os pontos de uma vez
        colors = centroids_plot[idx_sampled]
        ax.scatter(
            X_sampled[:, 0],
            X_sampled[:, 1],
            X_sampled[:, 2],
            c=colors,
            s=2,
            alpha=0.4
        )
        
        # Plotar centróides
        ax.scatter(
            centroids_plot[:, 0],
            centroids_plot[:, 1],
            centroids_plot[:, 2],
            c=centroids_plot,
            s=150,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            alpha=1.0
        )
        
        ax.set_xlabel('Red', fontsize=8)
        ax.set_ylabel('Green', fontsize=8)
        ax.set_zlabel('Blue', fontsize=8)
        ax.set_title(f'K={K}', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=7)
    
    plt.suptitle('K-Means no Espaço RGB', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Concluído!\n")


def plot_all_palettes(results):
    """
    Mostra paletas de cores para todos os resultados.
    
    Args:
        results: Lista de resultados
    """
    for result in results:
        K = result['K']
        print(f"\n--- Paleta para K={K} ---")
        show_centroid_colors(result['centroids'])


def plot_all_zooms(results, zoom_size=200):
    """
    Plota zoom comparativo para todos os resultados.
    
    Args:
        results: Lista de resultados
        zoom_size: Tamanho da região de zoom
    """
    for result in results:
        plot_zoom_comparison(
            result['original_img'],
            result['compressed_img'],
            result['K'],
            zoom_size=zoom_size,
            seed=42
        )