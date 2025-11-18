"""Visualizações para K-Means em compressão de imagens."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# importa funções que medem o tamanho REAL dos PNGs em memória (equivalente ao arquivo)
from .utils import (
    compute_png_size_rgb_mb,
    compute_png_size_palette_mb
)


def normalize_colors(array):
    """Normaliza array para [0,1] se necessário."""
    return array / 255.0 if array.max() > 1.0 else array


def to_float(img):
    """Converte imagem para float32 normalizado."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


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
        ax.set_title(f'Cor {i}', fontsize=12)

    plt.suptitle("Paleta de Cores", fontsize=14, y=0.85)
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
        print("⚠️ Imagem muito pequena para zoom")
        return

    if seed is not None:
        np.random.seed(seed)

    y = np.random.randint(0, H - zh)
    x = np.random.randint(0, W - zw)

    zoom_orig = original[y:y+zh, x:x+zw]
    zoom_comp = compressed[y:y+zh, x:x+zw]

    # Métricas de cores
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
    ax3.set_title('Zoom Original', fontsize=14, fontweight='bold')
    ax3.axis('off')

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(zoom_comp)
    ax4.set_title('Zoom Comprimida', fontsize=14, fontweight='bold')
    ax4.axis('off')

    if zoom_size <= 50:
        for ax in [ax3, ax4]:
            ax.set_xticks(np.arange(-0.5, zw, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, zh, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-',
                    linewidth=0.5, alpha=0.3)

    plt.suptitle(
        f'Zoom - K={K}\nCores: {colors_o} → {colors_c}',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    plt.show()

    print(f"\n🔍 ZOOM: ({x}, {y}) - {zw}×{zh}px | Cores: {colors_o} → {colors_c}\n")


def plot_comparison(result):
    """
    Plota comparação entre:
      - JPEG original (tamanho do arquivo .jpg)
      - PNG RGB (imagem original convertida para PNG)
      - PNG paleta (imagem quantizada pelo K-Means)

    Mostrando em cada título:
      - Tamanho em disco (arquivo)
      - Tamanho em memória (representação crua ou paleta+índice)

    Args:
        result: Dict com:
            - 'original_img': imagem original (RGB)
            - 'compressed_img': imagem quantizada (RGB)
            - 'cores_originais', 'cores_comprimidas'
            - 'K'
            - opcional: 'jpeg_size_mb' (tamanho do .jpg em MB)
            - opcional: 'tamanho_original_MB' (RGB cru, H*W*3)
            - opcional: 'tamanho_comprimido_MB' (paleta+idx teórico)
    """
    orig = result['original_img']
    comp = result['compressed_img']
    K = result['K']

    colors_o = result['cores_originais']
    colors_c = result['cores_comprimidas']

    # Tamanho do JPEG (se fornecido)
    jpeg_mb = result.get('jpeg_size_mb', None)

    # Tamanhos REAIS em PNG (arquivo), sem optimize
    png_rgb_mb = compute_png_size_rgb_mb(orig, optimize=False)
    png_pal_mb = compute_png_size_palette_mb(result, optimize=False)

    # Tamanhos em memória (representações internas)
    mem_rgb_mb = result.get('tamanho_original_MB', None)      # RGB cru
    mem_pal_mb = result.get('tamanho_comprimido_MB', None)    # paleta+idx

    # helpers de string
    def fmt_mem_rgb():
        return f"{mem_rgb_mb:.2f} MB" if mem_rgb_mb is not None else "N/A"

    def fmt_mem_pal():
        return f"{mem_pal_mb:.2f} MB" if mem_pal_mb is not None else "N/A"

    # Plot 1x3: JPEG | PNG RGB | PNG paleta
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    # 1) JPEG original (mesmo conteúdo visual do orig)
    ax[0].imshow(orig)
    if jpeg_mb is not None:
        title_jpeg = (
            f'JPEG Original\n'
            f'Disco: {jpeg_mb:.2f} MB | Memória RGB: {fmt_mem_rgb()}\n'
            f'{colors_o:,} cores'
        )
    else:
        title_jpeg = (
            f'JPEG Original\n'
            f'Disco: (não informado) | Memória RGB: {fmt_mem_rgb()}\n'
            f'{colors_o:,} cores'
        )
    ax[0].set_title(title_jpeg, fontsize=13)
    ax[0].axis('off')

    # 2) PNG RGB (sem quantização de cor)
    title_png_rgb = (
        f'PNG RGB (sem K-Means)\n'
        f'Disco: {png_rgb_mb:.2f} MB | Memória RGB: {fmt_mem_rgb()}\n'
        f'{colors_o:,} cores'
    )
    ax[1].imshow(orig)
    ax[1].set_title(title_png_rgb, fontsize=13)
    ax[1].axis('off')

    # 3) PNG paleta (K-Means)
    title_png_pal = (
        f'PNG paleta (K={K})\n'
        f'Disco: {png_pal_mb:.2f} MB | Memória paleta+idx: {fmt_mem_pal()}\n'
        f'{colors_c:,} cores'
    )
    ax[2].imshow(comp)
    ax[2].set_title(title_png_pal, fontsize=13)
    ax[2].axis('off')

    # Supertítulo
    plt.suptitle('Comparação de Formatos', fontsize=16, y=0.97)

    plt.tight_layout()
    plt.show()

    print("\n📦 Tamanhos reais de ARQUIVO:")
    if jpeg_mb is not None:
        print(f"   JPEG original        : {jpeg_mb:.3f} MB")
    else:
        print("   JPEG original        : (tamanho não informado)")
    print(f"   PNG RGB (sem K-Means): {png_rgb_mb:.3f} MB")
    print(f"   PNG paleta (K={K})   : {png_pal_mb:.3f} MB")

    print("\n🧠 Tamanhos em MEMÓRIA (representação interna):")
    if mem_rgb_mb is not None:
        print(f"   RGB bruto (H*W*3)       : {mem_rgb_mb:.3f} MB")
    else:
        print("   RGB bruto (H*W*3)       : (não informado)")
    if mem_pal_mb is not None:
        print(f"   Paleta+índice (K-Means) : {mem_pal_mb:.3f} MB")
    else:
        print("   Paleta+índice (K-Means) : (não informado)")
    print("")


def plot_result(result, show_comparison=True, show_colors=False,
                show_zoom=False, zoom_size=200):
    """
    Plota visualizações para um resultado.

    Args:
        result: Dict de resultado do compress_image
        show_comparison: Comparação JPEG vs PNG RGB vs PNG paleta
        show_colors: Paleta de cores
        show_zoom: Zoom comparativo
        zoom_size: Tamanho do zoom
    """
    K = result['K']
    print(f"\n{'='*70}")
    print(f"VISUALIZANDO K={K}")
    print(f"{'='*70}\n")

    if show_comparison:
        plot_comparison(result)

    if show_colors:
        show_palette(result['centroids'])

    if show_zoom:
        plot_zoom(result.get('original_img', result['compressed_img']),
                  result['compressed_img'], K, zoom_size, seed=42)