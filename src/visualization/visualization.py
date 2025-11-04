"""
Funções de visualização para resultados de K-Means.
Use após processar com run_kmeans_grid.
"""

from src.visualization.plot_3d import (
    plot_kmeans_rgb,
    show_centroid_colors,
    print_compression_analysis,
    print_compression_comparison,
    plot_zoom_comparison
)
import matplotlib.pyplot as plt


# ========== PLOT INDIVIDUAL ==========

def plot_single_result(result, show_comparison=True, show_rgb=False,
                       show_palette=False, show_analysis=False,
                       show_zoom=False, zoom_size=200):
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
            K
        )
    
    # 3. Paleta de cores
    if show_palette:
        show_centroid_colors(result['centroids'])
    
    # 4. Análise de compressão
    if show_analysis:
        print_compression_analysis(
            result['original_img'].shape,
            result['centroids'],
            result['idx'],
            K
        )
    
    # 5. Zoom comparativo
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


def plot_all_rgb(results):
    """
    Plota visualizações RGB 3D para todos os resultados.
    
    Args:
        results: Lista de resultados
    """
    for result in results:
        K = result['K']
        print(f"\n--- RGB 3D para K={K} ---")
        plot_kmeans_rgb(
            result['X'],
            result['centroids'],
            result['idx'],
            K
        )


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


def print_all_analyses(results):
    """
    Imprime análise de compressão para todos os resultados.
    
    Args:
        results: Lista de resultados
    """
    for result in results:
        print_compression_analysis(
            result['original_img'].shape,
            result['centroids'],
            result['idx'],
            result['K']
        )


def print_summary(results):
    """
    Imprime resumo comparativo de todos os resultados.
    
    Args:
        results: Lista de resultados
    """
    print_compression_comparison(results)


# ========== GRID DE COMPARAÇÕES ==========

def plot_comparison_grid(results, max_cols=3):
    """
    Plota grid de comparações para múltiplos K.
    
    Args:
        results: Lista de resultados
        max_cols: Máximo de colunas no grid
    """
    n = len(results)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 8, rows * 4))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        row = i // cols
        col = (i % cols) * 2
        
        K = result['K']
        original = result['original_img']
        compressed = result['compressed_img']
        psnr = result['PSNR_dB']
        ratio = result['fator_compactacao']
        
        # Original
        axes[row, col].imshow(original)
        axes[row, col].set_title(f'Original', fontsize=10)
        axes[row, col].axis('off')
        
        # Comprimida
        axes[row, col + 1].imshow(compressed)
        axes[row, col + 1].set_title(
            f'K={K} | {ratio:.1f}x | {psnr:.1f}dB',
            fontsize=10
        )
        axes[row, col + 1].axis('off')
    
    # Esconder eixos vazios
    for i in range(n, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.suptitle('Comparação de Múltiplos K', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()