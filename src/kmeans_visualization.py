"""Visualizações para K-Means em compressão de imagens."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def normalize_colors(array):
    """Normaliza array para [0,1] se necessário."""
    return array / 255.0 if array.max() > 1.0 else array


def to_float(img):
    """Converte imagem para float32 normalizado."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def plot_kmeans_rgb(X, centroids, idx, K, max_points=10000):
    """
    Plota K-Means no espaço RGB 3D com subsampling.
    
    Args:
        X: Dados (n_samples, 3)
        centroids: Centróides (K, 3)
        idx: Índices dos clusters (n_samples,)
        K: Número de clusters
        max_points: Máximo de pontos a plotar
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X_plot = normalize_colors(X)
    centroids_plot = normalize_colors(centroids)
    
    n_samples = X_plot.shape[0]
    
    if n_samples > max_points:
        indices = np.random.choice(n_samples, max_points, replace=False)
        X_plot = X_plot[indices]
        idx = idx[indices]
        print(f"   ⚡ {n_samples:,} → {max_points:,} pontos")
    
    colors = centroids_plot[idx]
    ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
               c=colors, s=3, alpha=0.5)
    
    ax.scatter(centroids_plot[:, 0], centroids_plot[:, 1],
               centroids_plot[:, 2], c=centroids_plot, s=200,
               marker='*', edgecolors='black', linewidths=2,
               alpha=1.0, label='Centróides')
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(f'K-Means no espaço RGB (K={K})', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


def show_palette(centroids):
    """
    Mostra paleta de cores dos centróides.
    
    Args:
        centroids: Centróides (K, 3)
    """
    K = centroids.shape[0]
    colors = normalize_colors(centroids)
    
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


def plot_zoom(original, compressed, K, zoom_size=200, seed=None):
    """
    Plota zoom comparando original vs comprimida.
    
    Args:
        original: Imagem original
        compressed: Imagem comprimida
        K: Número de cores
        zoom_size: Tamanho da região de zoom
        seed: Seed para posição aleatória
    """
    H, W = original.shape[:2]
    zh, zw = min(zoom_size, H), min(zoom_size, W)
    
    if zh <= 0 or zw <= 0 or H <= zh or W <= zw:
        print("⚠️  Imagem muito pequena para zoom")
        return
    
    if seed is not None:
        np.random.seed(seed)
    
    y = np.random.randint(0, H - zh)
    x = np.random.randint(0, W - zw)
    
    zoom_orig = original[y:y+zh, x:x+zw]
    zoom_comp = compressed[y:y+zh, x:x+zw]
    
    # Métricas
    orig_f = to_float(zoom_orig)
    comp_f = to_float(zoom_comp)
    
    mse = np.mean((orig_f - comp_f) ** 2)
    psnr = -10.0 * np.log10(mse) if mse > 0 else float('inf')
    
    colors_o = len(np.unique(zoom_orig.reshape(-1, 3), axis=0))
    colors_c = len(np.unique(zoom_comp.reshape(-1, 3), axis=0))
    
    # Plot
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original)
    ax1.set_title('Original - Completa', fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax1.add_patch(Rectangle((x, y), zw, zh, linewidth=3,
                            edgecolor='red', facecolor='none'))
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(compressed)
    ax2.set_title(f'Comprimida (K={K}) - Completa',
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax2.add_patch(Rectangle((x, y), zw, zh, linewidth=3,
                            edgecolor='red', facecolor='none'))
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(zoom_orig)
    ax3.set_title('Zoom Original', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(zoom_comp)
    ax4.set_title('Zoom Comprimida', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    if zoom_size <= 50:
        for ax in [ax3, ax4]:
            ax.set_xticks(np.arange(-0.5, zw, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, zh, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-',
                    linewidth=0.5, alpha=0.3)
    
    reduction = (1 - colors_c / colors_o) * 100 if colors_o > 0 else 0
    plt.suptitle(
        f'Zoom - K={K}\n'
        f'PSNR={psnr:.2f} dB | Cores: {colors_o} → {colors_c} '
        f'({reduction:.1f}% redução)',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔍 ZOOM: ({x}, {y}) - {zw}×{zh}px | "
          f"PSNR: {psnr:.2f} dB | "
          f"Cores: {colors_o} → {colors_c}\n")


def plot_comparison(result):
    """
    Plota comparação lado-a-lado.
    
    Args:
        result: Dict com 'original_img', 'compressed_img', etc.
    """
    orig = result['original_img']
    comp = result['compressed_img']
    K = result['K']
    
    colors_o = result['cores_originais']
    colors_c = result['cores_comprimidas']
    size_o = result['tamanho_original_MB']
    size_c = result['tamanho_comprimido_MB']
    ratio = result['fator_compactacao']
    psnr = result['PSNR_dB']
    
    reduction = (1 - size_c / size_o) * 100
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(orig)
    ax[0].set_title(f'Original\n{size_o:.2f} MB | {colors_o:,} cores',
                    fontsize=14)
    ax[0].axis('off')
    
    ax[1].imshow(comp)
    ax[1].set_title(f'Comprimida (K={K})\n'
                    f'{size_c:.2f} MB | {colors_c:,} cores',
                    fontsize=14)
    ax[1].axis('off')
    
    plt.suptitle(
        f'Compressão {ratio:.2f}x ({reduction:.1f}% redução) | '
        f'PSNR: {psnr:.1f} dB',
        fontsize=16, y=0.95
    )
    
    plt.tight_layout()
    plt.show()


def plot_result(result, show_comparison=True, show_rgb=False,
                show_palette=False, show_zoom=False, zoom_size=200,
                max_points=10000):
    """
    Plota visualizações para um resultado.
    
    Args:
        result: Dict de resultado do compress_image
        show_comparison: Comparação lado-a-lado
        show_rgb: Plot 3D RGB
        show_palette: Paleta de cores
        show_zoom: Zoom comparativo
        zoom_size: Tamanho do zoom
        max_points: Máximo de pontos para RGB 3D
    """
    K = result['K']
    print(f"\n{'='*70}")
    print(f"VISUALIZANDO K={K}")
    print(f"{'='*70}\n")
    
    if show_comparison:
        plot_comparison(result)
    
    if show_rgb:
        X = result['compressed_img'].reshape(-1, 3)
        plot_kmeans_rgb(X, result['centroids'], result['idx'],
                        K, max_points)
    
    if show_palette:
        show_palette(result['centroids'])
    
    if show_zoom:
        plot_zoom(result.get('original_img', result['compressed_img']),
                  result['compressed_img'], K, zoom_size, seed=42)