import matplotlib.pyplot as plt
import numpy as np


def plot_kMeans_RGB(X, centroids, idx, K):
    """
    Plota o resultado do K-Means no espaço RGB.
    Cada pixel é colorido com a cor do seu centróide.
    (Sem marcadores de centróide nem legenda.)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normaliza para [0,1] se estiver em [0,255]
    if X.max() > 1.0 or centroids.max() > 1.0:
        X_plot = X / 255.0
        centroids_plot = centroids / 255.0
    else:
        X_plot = X
        centroids_plot = centroids

    # Plota os pixels com a cor do centróide correspondente
    for k in range(K):
        cluster_points = X_plot[idx == k]
        if cluster_points.size == 0:
            continue
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=[centroids_plot[k]],  # cor real do centróide
            s=5,
            alpha=0.6
        )

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('K-Means no espaço RGB (cores = centróides)', fontsize=10)
    plt.tight_layout()
    plt.show()


def show_centroid_colors(centroids):
    """
    Mostra as cores representadas pelos centróides do K-Means (paleta).
    """
    num_centroids = centroids.shape[0]

    # Normaliza para [0,1] se necessário
    if centroids.max() > 1.0:
        centroids = centroids / 255.0

    fig, ax = plt.subplots(1, num_centroids, figsize=(num_centroids * 2, 2))
    if num_centroids == 1:
        ax = [ax]

    for i in range(num_centroids):
        ax[i].imshow([[centroids[i]]])
        ax[i].axis('off')
        ax[i].set_title(f'Idx {i}', fontsize=8)

    plt.suptitle("Cores dos centróides (paleta)", y=0.9, fontsize=10)
    plt.show()


def print_compression_analysis(original_shape, centroids, idx, K):
    """
    Imprime análise DETALHADA da compressão mostrando por que
    K=4 e K=128 têm o mesmo tamanho (ambos usam uint8).
    
    Parâmetros:
        original_shape: (H, W, C) da imagem original
        centroids: array de centróides
        idx: array de índices
        K: número de clusters
    """
    H, W, C = original_shape
    n_pixels = H * W
    
    # Determinar dtype ótimo para índices
    if K <= 256:
        idx_dtype = np.uint8
        idx_bytes_per_pixel = 1
        uint_range = "0 a 255"
    elif K <= 65536:
        idx_dtype = np.uint16
        idx_bytes_per_pixel = 2
        uint_range = "0 a 65,535"
    else:
        idx_dtype = np.uint32
        idx_bytes_per_pixel = 4
        uint_range = "0 a 4,294,967,295"
    
    # Calcular tamanhos
    # Original
    original_bytes = n_pixels * 3 * 1  # RGB uint8
    original_mb = original_bytes / (1024 * 1024)
    
    # Comprimida
    centroids_bytes = K * 3 * 4  # float32
    centroids_kb = centroids_bytes / 1024
    centroids_mb = centroids_bytes / (1024 * 1024)
    
    indices_bytes = n_pixels * idx_bytes_per_pixel
    indices_mb = indices_bytes / (1024 * 1024)
    
    compressed_bytes = centroids_bytes + indices_bytes
    compressed_mb = compressed_bytes / (1024 * 1024)
    
    # Percentuais
    pct_indices = (indices_bytes / compressed_bytes) * 100
    pct_centroids = (centroids_bytes / compressed_bytes) * 100
    
    # Taxa de compressão
    compression_ratio = original_mb / compressed_mb if compressed_mb > 0 else float('inf')
    savings_pct = (1 - compressed_mb / original_mb) * 100
    
    # Imprimir análise
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISE DETALHADA DA COMPRESSÃO (K={K})")
    print(f"{'='*70}")
    
    print(f"\n📐 IMAGEM ORIGINAL:")
    print(f"   • Resolução: {H} × {W} = {n_pixels:,} pixels")
    print(f"   • Formato: RGB uint8 (3 canais)")
    print(f"   • Bytes por pixel: 3 bytes (R + G + B)")
    print(f"   • Tamanho total: {original_mb:.2f} MB ({original_bytes:,} bytes)")
    
    print(f"\n🗜️  IMAGEM COMPRIMIDA (K={K}):")
    print(f"\n   📦 Centróides (paleta de cores):")
    print(f"      ├─ Quantidade: {K} cores")
    print(f"      ├─ Formato: RGB float32 (3 canais × 4 bytes = 12 bytes/cor)")
    print(f"      ├─ Cálculo: {K} cores × 12 bytes = {centroids_bytes:,} bytes")
    print(f"      ├─ Tamanho: {centroids_kb:.2f} KB ({centroids_mb:.4f} MB)")
    print(f"      └─ Percentual: {pct_centroids:.3f}% do total comprimido")
    
    print(f"\n   🗺️  Índices (mapa de pixels → cores):")
    print(f"      ├─ Tipo: {idx_dtype.__name__} ({idx_bytes_per_pixel} byte por pixel)")
    print(f"      ├─ Range: 0 a {K-1} (cabe em {uint_range})")
    print(f"      ├─ Cálculo: {n_pixels:,} pixels × {idx_bytes_per_pixel} byte = {indices_bytes:,} bytes")
    print(f"      ├─ Tamanho: {indices_mb:.2f} MB")
    print(f"      └─ Percentual: {pct_indices:.2f}% do total comprimido ⬅️ DOMINANTE!")
    
    print(f"\n   💾 Total comprimido: {compressed_mb:.2f} MB")
    
    print(f"\n📊 RESULTADO:")
    print(f"   • Taxa de compressão: {compression_ratio:.2f}x")
    print(f"   • {"Economia" if savings_pct > 0 else "Aumento"}: {abs(savings_pct):.1f}%")
    print(f"   • Bytes economizados: {(original_bytes - compressed_bytes):,}")
    print(f"   • Bytes/pixel original: 3.000")
    print(f"   • Bytes/pixel comprimido: {(compressed_bytes/n_pixels):.3f}")
    
    print(f"\n💡 POR QUE {idx_dtype.__name__.upper()}?")
    print(f"\n   Para representar índices de 0 a {K-1}, precisamos de:")
    print(f"   ")
    print(f"   • uint8:  0 a 255           (1 byte)   {'✅ USADO - Suficiente e eficiente!' if K <= 256 else '❌ Insuficiente'}")
    print(f"   • uint16: 0 a 65,535        (2 bytes)  {'✅ USADO - Mínimo necessário' if 256 < K <= 65536 else ('❌ Insuficiente' if K > 65536 else '⚠️  Desperdício (usa 2x mais memória)')}")
    print(f"   • uint32: 0 a 4,294,967,295 (4 bytes)  {'✅ USADO - Mínimo necessário' if K > 65536 else '⚠️  Desperdício (usa 4x mais memória)'}")
    
    print(f"\n🔑 INSIGHT IMPORTANTE:")
    print(f"\n   Os ÍNDICES dominam o tamanho ({pct_indices:.1f}%)!")
    print(f"   Os centróides são DESPREZÍVEIS ({pct_centroids:.2f}%).")
    print(f"   ")
    print(f"   Por isso:")
    print(f"   • K=4 usa ~{4*12} bytes em centróides")
    print(f"   • K=128 usa ~{128*12} bytes em centróides")
    print(f"   • Diferença: apenas {abs(128*12 - 4*12)} bytes = {abs(128*12 - 4*12)/1024:.2f} KB!")
    print(f"   ")
    print(f"   Ambos têm PRATICAMENTE O MESMO tamanho porque:")
    print(f"   ✅ Ambos usam uint8 (1 byte/pixel) para índices")
    print(f"   ✅ Índices representam >{pct_indices:.0f}% do tamanho")
    print(f"   ✅ Centróides são <{pct_centroids:.1f}% (insignificante!)")
    print(f"   ")
    print(f"   O tamanho só muda significativamente quando:")
    print(f"   🔄 K > 256 → uint16 (2 bytes) → TAMANHO DOBRA!")
    print(f"   🔄 K > 65,536 → uint32 (4 bytes) → TAMANHO DOBRA DE NOVO!")
    
    print(f"\n{'='*70}\n")
    
    return {
        'original_mb': original_mb,
        'compressed_mb': compressed_mb,
        'compression_ratio': compression_ratio,
        'savings_pct': savings_pct,
        'idx_dtype': idx_dtype.__name__,
        'bytes_per_pixel_original': 3,
        'bytes_per_pixel_compressed': compressed_bytes / n_pixels,
        'pct_indices': pct_indices,
        'pct_centroids': pct_centroids
    }


def print_compression_comparison(results_list):
    """
    Imprime tabela comparativa de múltiplos resultados (diferentes K).
    Mostra claramente quando o uint muda e o impacto no tamanho.
    
    Parâmetros:
        results_list: lista de dicts com resultados de run_kmeans_grid
    """
    import pandas as pd
    
    df = pd.DataFrame(results_list)
    
    print(f"\n{'='*100}")
    print(f"📊 COMPARAÇÃO DE COMPRESSÃO - MÚLTIPLOS K")
    print(f"{'='*100}\n")
    
    # Cabeçalho da tabela
    print(f"{'K':<6} {'uint':<8} {'B/px':<6} {'Centróides':<15} {'Índices':<15} {'Total':<12} {'Compressão':<12} {'PSNR':<10}")
    print(f"{'-'*6} {'-'*8} {'-'*6} {'-'*15} {'-'*15} {'-'*12} {'-'*12} {'-'*10}")
    
    previous_uint = None
    
    for _, row in df.iterrows():
        K = row['K']
        dtype_name = row['idx_dtype']
        bytes_pp = 1 if K <= 256 else (2 if K <= 65536 else 4)
        
        # Calcular componentes
        cent_kb = (K * 3 * 4) / 1024
        idx_mb = row['tamanho_comprimido_MB'] - (cent_kb / 1024)
        total_mb = row['tamanho_comprimido_MB']
        ratio = row['fator_compactacao']
        psnr = row['PSNR_dB']
        
        # Detectar mudança de uint
        if previous_uint is not None and previous_uint != dtype_name:
            print(f"{'─'*6} {'─'*8} {'─'*6} {'─'*15} {'─'*15} {'─'*12} {'─'*12} {'─'*10}")
            print(f"{'⚠️  MUDANÇA DE UINT! Tamanho dobra aqui ⬆️':^100}")
            print(f"{'─'*6} {'─'*8} {'─'*6} {'─'*15} {'─'*15} {'─'*12} {'─'*12} {'─'*10}")
        
        # Imprimir linha
        print(f"{K:<6} {dtype_name:<8} {bytes_pp:<6} {cent_kb:>10.2f} KB   {idx_mb:>10.2f} MB   {total_mb:>8.2f} MB   {ratio:>8.2f}x      {psnr:>6.2f} dB")
        
        previous_uint = dtype_name
    
    print(f"\n{'='*100}")
    
    # Análise geral
    print(f"\n💡 OBSERVAÇÕES IMPORTANTES:")
    print(f"\n1. 🟢 UINT8 (K ≤ 256):")
    uint8_rows = df[df['K'] <= 256]
    if len(uint8_rows) > 0:
        min_size = uint8_rows['tamanho_comprimido_MB'].min()
        max_size = uint8_rows['tamanho_comprimido_MB'].max()
        print(f"   • Todos têm tamanhos MUITO similares: {min_size:.2f} - {max_size:.2f} MB")
        print(f"   • Diferença máxima: apenas {(max_size - min_size)*1024:.1f} KB!")
        print(f"   • Isso acontece porque índices dominam (>99% do tamanho)")
        print(f"   • Centróides são desprezíveis (<1%)")
    
    print(f"\n2. 🟡 UINT16 (K = 257-65,536):")
    uint16_rows = df[(df['K'] > 256) & (df['K'] <= 65536)]
    if len(uint16_rows) > 0:
        avg_size = uint16_rows['tamanho_comprimido_MB'].mean()
        print(f"   • Tamanho médio: ~{avg_size:.2f} MB")
        print(f"   • Aproximadamente 2x maior que uint8")
        print(f"   • Usa 2 bytes por pixel ao invés de 1")
    
    print(f"\n3. 🔴 UINT32 (K > 65,536):")
    uint32_rows = df[df['K'] > 65536]
    if len(uint32_rows) > 0:
        avg_size = uint32_rows['tamanho_comprimido_MB'].mean()
        print(f"   • Tamanho médio: ~{avg_size:.2f} MB")
        print(f"   • Aproximadamente 4x maior que uint8")
        print(f"   • Usa 4 bytes por pixel")
    
    # Recomendações
    print(f"\n🎯 RECOMENDAÇÕES:")
    print(f"\n   Para MELHOR COMPRESSÃO:")
    print(f"   ✅ Use K ≤ 256 (uint8) - máxima eficiência de espaço")
    print(f"   ✅ Dentro desse range, prefira K maiores (ex: K=128 ou K=256)")
    print(f"   ✅ Você ganha qualidade SEM aumentar o tamanho!")
    print(f"   ")
    print(f"   Evite:")
    print(f"   ❌ K > 256 a menos que qualidade seja MUITO mais importante")
    print(f"   ❌ O salto de K=256 para K=257 dobra o tamanho!")
    
    # Melhor escolha
    best_k = df.loc[df['PSNR_dB'].idxmax()]
    best_compression = df.loc[df['fator_compactacao'].idxmax()]
    
    print(f"\n📈 DESTAQUES:")
    print(f"   • Melhor qualidade: K={best_k['K']} (PSNR={best_k['PSNR_dB']:.2f} dB, {best_k['tamanho_comprimido_MB']:.2f} MB)")
    print(f"   • Melhor compressão: K={best_compression['K']} ({best_compression['fator_compactacao']:.2f}x, {best_compression['tamanho_comprimido_MB']:.2f} MB)")
    
    print(f"\n{'='*100}\n")

def plot_zoom_comparison(original_img, compressed_img, K, zoom_size=200, seed=None):
    """
    Plota zoom em região aleatória comparando original vs comprimida.
    Mostra os efeitos da quantização em detalhes.
    
    Parâmetros:
        original_img: imagem original
        compressed_img: imagem comprimida
        K: número de cores usado
        zoom_size: tamanho da região de zoom (pixels)
        seed: seed para posição aleatória (None = aleatório)
    """
    H, W, C = original_img.shape
    
    # Garantir que zoom_size não seja maior que a imagem
    zoom_h = min(zoom_size, H)
    zoom_w = min(zoom_size, W)
    
    # Escolher posição aleatória para zoom
    if seed is not None:
        np.random.seed(seed)
    
    # Garantir que a região caiba na imagem
    max_y = H - zoom_h
    max_x = W - zoom_w
    
    if max_y <= 0 or max_x <= 0:
        print("⚠️  Imagem muito pequena para zoom")
        return
    
    start_y = np.random.randint(0, max_y)
    start_x = np.random.randint(0, max_x)
    
    end_y = start_y + zoom_h
    end_x = start_x + zoom_w
    
    # Extrair regiões
    zoom_original = original_img[start_y:end_y, start_x:end_x]
    zoom_compressed = compressed_img[start_y:end_y, start_x:end_x]
    
    # Criar figura
    fig = plt.figure(figsize=(20, 10))
    
    # ========== ROW 1: Imagens completas com retângulo ==========
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original - Imagem Completa', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # ========== ROW 2: Zoom nas regiões ==========
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(zoom_original)
    ax3.set_title(f'Zoom Original', 
                  fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Adicionar grid para ver pixels individuais se zoom for pequeno
    if zoom_size <= 50:
        ax3.set_xticks(np.arange(-0.5, zoom_w, 1), minor=True)
        ax3.set_yticks(np.arange(-0.5, zoom_h, 1), minor=True)
        ax3.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(zoom_compressed)
    ax4.set_title(f'Zoom Comprimida', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Adicionar grid
    if zoom_size <= 50:
        ax4.set_xticks(np.arange(-0.5, zoom_w, 1), minor=True)
        ax4.set_yticks(np.arange(-0.5, zoom_h, 1), minor=True)
        ax4.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Calcular diferença na região de zoom
    def to_float(img):
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
    zoom_orig_f = to_float(zoom_original)
    zoom_comp_f = to_float(zoom_compressed)
    
    # Métricas da região de zoom
    mse_zoom = np.mean((zoom_orig_f - zoom_comp_f) ** 2)
    if mse_zoom > 0:
        psnr_zoom = 20 * np.log10(1.0) - 10 * np.log10(mse_zoom)
    else:
        psnr_zoom = float('inf')
    
    # Cores únicas na região
    colors_orig_zoom = len(np.unique(zoom_original.reshape(-1, 3), axis=0))
    colors_comp_zoom = len(np.unique(zoom_compressed.reshape(-1, 3), axis=0))
    
    # Título geral
    plt.suptitle(
        f'Comparação com Zoom - K={K} cores\n'
        f'Região de Zoom: PSNR={psnr_zoom:.2f} dB | '
        f'Cores: {colors_orig_zoom} → {colors_comp_zoom} | '
        f'Redução: {((1 - colors_comp_zoom/colors_orig_zoom) * 100):.1f}%',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir informações
    print(f"\n🔍 ANÁLISE DA REGIÃO DE ZOOM:")
    print(f"   Posição: ({start_x}, {start_y}) até ({end_x}, {end_y})")
    print(f"   Tamanho: {zoom_w}×{zoom_h} pixels")
    print(f"   PSNR da região: {psnr_zoom:.2f} dB")
    print(f"   Cores originais: {colors_orig_zoom}")
    print(f"   Cores comprimidas: {colors_comp_zoom}")
    print(f"   Redução de cores: {((1 - colors_comp_zoom/colors_orig_zoom) * 100):.1f}%\n")