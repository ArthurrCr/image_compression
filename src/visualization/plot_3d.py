"""
Funções de visualização para K-Means em compressão de imagens.
Inclui plots 3D, paletas de cores, análises de compressão e zoom.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ========== VISUALIZAÇÕES 3D E PALETA ==========

def plot_kmeans_rgb(X, centroids, idx, K):
    """
    Plota resultado do K-Means no espaço RGB 3D.
    
    Args:
        X: Dados (n_samples, 3)
        centroids: Centróides (K, 3)
        idx: Índices dos clusters (n_samples,)
        K: Número de clusters
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalizar para [0,1]
    X_plot = X / 255.0 if X.max() > 1.0 else X
    centroids_plot = centroids / 255.0 if centroids.max() > 1.0 else centroids
    
    # Plotar cada cluster com cor do centróide
    for k in range(K):
        cluster_points = X_plot[idx == k]
        if cluster_points.size == 0:
            continue
        
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=[centroids_plot[k]],
            s=5,
            alpha=0.6
        )
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('K-Means no espaço RGB', fontsize=12)
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


def print_compression_analysis(original_shape, centroids, idx, K):
    """
    Imprime análise detalhada da compressão.
    
    Args:
        original_shape: (H, W, C) da imagem original
        centroids: Array de centróides
        idx: Array de índices
        K: Número de clusters
        
    Returns:
        Dict com estatísticas de compressão
    """
    H, W, C = original_shape
    n_pixels = H * W
    
    # Informações sobre dtype
    idx_dtype, bytes_per_pixel, uint_range = get_index_dtype_info(K)
    
    # Tamanhos
    original_bytes = n_pixels * 3  # RGB uint8
    original_mb = original_bytes / (1024 * 1024)
    
    centroids_bytes = K * 3 * 4  # float32
    centroids_kb = centroids_bytes / 1024
    
    indices_bytes = n_pixels * bytes_per_pixel
    indices_mb = indices_bytes / (1024 * 1024)
    
    compressed_bytes = centroids_bytes + indices_bytes
    compressed_mb = compressed_bytes / (1024 * 1024)
    
    # Percentuais
    pct_indices = (indices_bytes / compressed_bytes) * 100
    pct_centroids = (centroids_bytes / compressed_bytes) * 100
    
    # Taxa de compressão
    ratio = original_mb / compressed_mb if compressed_mb > 0 else float('inf')
    savings_pct = (1 - compressed_mb / original_mb) * 100
    
    # Imprimir
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISE DE COMPRESSÃO (K={K})")
    print(f"{'='*70}")
    
    print(f"\n📐 ORIGINAL:")
    print(f"   • Resolução: {H}×{W} ({n_pixels:,} pixels)")
    print(f"   • Tamanho: {original_mb:.2f} MB (RGB uint8)")
    
    print(f"\n🗜️  COMPRIMIDA:")
    print(f"   • Centróides: {centroids_kb:.2f} KB ({pct_centroids:.2f}%)")
    print(f"   • Índices: {indices_mb:.2f} MB ({pct_indices:.2f}%) ⬅️ Dominante")
    print(f"   • Total: {compressed_mb:.2f} MB")
    print(f"   • Dtype índices: {idx_dtype.__name__} ({bytes_per_pixel} byte/pixel)")
    
    print(f"\n📊 RESULTADO:")
    print(f"   • Taxa: {ratio:.2f}x")
    print(f"   • Economia: {savings_pct:.1f}%")
    print(f"   • Bytes/pixel: 3.0 → {compressed_bytes/n_pixels:.3f}")
    
    print(f"\n💡 INSIGHT:")
    print(f"   Índices dominam ({pct_indices:.0f}%)!")
    print(f"   K=4 e K=128 têm tamanho similar se ambos usam uint8.")
    print(f"   Tamanho só muda significativamente quando K > 256 (uint16).")
    
    print(f"\n{'='*70}\n")
    
    return {
        'original_mb': original_mb,
        'compressed_mb': compressed_mb,
        'compression_ratio': ratio,
        'savings_pct': savings_pct,
        'idx_dtype': idx_dtype.__name__,
        'pct_indices': pct_indices,
        'pct_centroids': pct_centroids
    }


def print_compression_comparison(results_list):
    """
    Imprime tabela comparativa de múltiplos K.
    
    Args:
        results_list: Lista de dicts com resultados
    """
    import pandas as pd
    
    df = pd.DataFrame(results_list)
    
    print(f"\n{'='*90}")
    print(f"📊 COMPARAÇÃO - MÚLTIPLOS K")
    print(f"{'='*90}\n")
    
    # Cabeçalho
    print(f"{'K':<6} {'uint':<8} {'B/px':<6} {'Centróides':<12} "
          f"{'Índices':<12} {'Total':<10} {'Taxa':<8} {'PSNR':<8}")
    print(f"{'-'*6} {'-'*8} {'-'*6} {'-'*12} "
          f"{'-'*12} {'-'*10} {'-'*8} {'-'*8}")
    
    previous_dtype = None
    
    for _, row in df.iterrows():
        K = row['K']
        dtype_name = row['idx_dtype']
        bytes_pp = 1 if K <= 256 else (2 if K <= 65536 else 4)
        
        cent_kb = (K * 3 * 4) / 1024
        idx_mb = row['tamanho_comprimido_MB'] - (cent_kb / 1024)
        total_mb = row['tamanho_comprimido_MB']
        ratio = row['fator_compactacao']
        psnr = row['PSNR_dB']
        
        # Detectar mudança de dtype
        if previous_dtype and previous_dtype != dtype_name:
            print(f"{'-'*90}")
            print(f"{'⚠️  MUDANÇA DE DTYPE - Tamanho dobra!':^90}")
            print(f"{'-'*90}")
        
        print(f"{K:<6} {dtype_name:<8} {bytes_pp:<6} "
              f"{cent_kb:>8.2f} KB  {idx_mb:>8.2f} MB  "
              f"{total_mb:>7.2f} MB  {ratio:>6.2f}x  {psnr:>6.1f} dB")
        
        previous_dtype = dtype_name
    
    print(f"\n{'='*90}")
    
    # Análise por categoria
    print(f"\n💡 OBSERVAÇÕES:")
    
    uint8_rows = df[df['K'] <= 256]
    if len(uint8_rows) > 0:
        min_size = uint8_rows['tamanho_comprimido_MB'].min()
        max_size = uint8_rows['tamanho_comprimido_MB'].max()
        print(f"\n🟢 UINT8 (K ≤ 256):")
        print(f"   • Tamanhos similares: {min_size:.2f} - {max_size:.2f} MB")
        print(f"   • Diferença: {(max_size - min_size)*1024:.1f} KB")
        print(f"   • Índices dominam (>99% do tamanho)")
    
    uint16_rows = df[(df['K'] > 256) & (df['K'] <= 65536)]
    if len(uint16_rows) > 0:
        avg_size = uint16_rows['tamanho_comprimido_MB'].mean()
        print(f"\n🟡 UINT16 (K = 257-65,536):")
        print(f"   • Tamanho médio: {avg_size:.2f} MB (~2x maior)")
    
    # Recomendações
    print(f"\n🎯 RECOMENDAÇÃO:")
    print(f"   Use K ≤ 256 para máxima eficiência!")
    print(f"   Prefira K maiores (128, 256) = melhor qualidade sem custo.")
    
    # Destaques
    best_quality = df.loc[df['PSNR_dB'].idxmax()]
    best_compression = df.loc[df['fator_compactacao'].idxmax()]
    
    print(f"\n📈 DESTAQUES:")
    print(f"   • Melhor qualidade: K={best_quality['K']} "
          f"(PSNR={best_quality['PSNR_dB']:.1f} dB)")
    print(f"   • Melhor compressão: K={best_compression['K']} "
          f"({best_compression['fator_compactacao']:.2f}x)")
    
    print(f"\n{'='*90}\n")


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