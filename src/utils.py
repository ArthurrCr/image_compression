import gc
import io
import os
import time
import numpy as np
from PIL import Image
import imageio.v2 as imageio

from .kmeans import kmeans_init_centroids, run_kmeans, clear_gpu_memory


def get_optimal_dtype(k):
    """Retorna dtype ótimo para índices."""
    if k <= 256:
        return np.uint8
    if k <= 65536:
        return np.uint16
    return np.uint32


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


def run_kmeans_single(
    X,
    k,
    max_iters=10,
    seed=0,
    n_init=1,
    use_gpu=True,
    batch_size=200000
):
    """
    Executa K-Means com múltiplas inicializações.
    Retorna o melhor resultado com base no SSE.
    """
    best_sse = float("inf")
    best_centroids = None
    best_idx = None

    for i in range(n_init):
        if seed is not None:
            np.random.seed(seed + i)

        initial_centroids = kmeans_init_centroids(X, k, use_gpu=use_gpu)

        centroids, idx = run_kmeans(
            X,
            initial_centroids,
            max_iters=max_iters,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

        sse = float(np.sum((X - centroids[idx]) ** 2))

        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids
            best_idx = idx

    return best_centroids, best_idx, best_sse


def make_palette_image(result):
    """
    Cria uma imagem indexada ('P') a partir do resultado do K-Means.

    Usa:
        - result['centroids']: paleta Kx3 em [0,1]
        - result['idx']: índices lineares
        - result['compressed_img']: para recuperar H e W
    """
    k = result["K"]

    if k > 256:
        raise ValueError(
            f"PNG com paleta suporta no máximo 256 cores, mas K={k}."
        )

    centroids = result["centroids"]
    idx = result["idx"]
    comp = result["compressed_img"]

    h, w = comp.shape[:2]

    idx_2d = idx.reshape(h, w).astype(np.uint8)
    palette = np.clip(np.rint(centroids * 255.0), 0, 255).astype(np.uint8)

    palette_list = palette.reshape(-1).tolist()
    palette_list += [0] * (256 * 3 - len(palette_list))

    img_p = Image.fromarray(idx_2d, mode="P")
    img_p.putpalette(palette_list)
    return img_p


def compute_png_size_rgb_mb(img, optimize=False):
    """
    Codifica imagem RGB como PNG em memória e retorna tamanho em MB.
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    pil_img = Image.fromarray(img_u8, mode="RGB")
    buf = io.BytesIO()

    pil_img.save(buf, format="PNG", optimize=optimize)

    size_bytes = len(buf.getvalue())
    buf.close()

    return size_bytes / (1024 * 1024)


def compute_png_size_palette_mb(result, optimize=False):
    """
    Codifica PNG-8 (paleta) gerado pelo K-Means e retorna tamanho em MB.
    """
    img_p = make_palette_image(result)
    buf = io.BytesIO()

    img_p.save(buf, format="PNG", optimize=optimize)

    size_bytes = len(buf.getvalue())
    buf.close()

    return size_bytes / (1024 * 1024)


def compress_image(
    img,
    k,
    max_iters=10,
    seed=0,
    n_init=1,
    use_gpu=True,
    batch_size=200000
):
    """
    Comprime imagem usando K-Means.

    Retorna:
        dict com resultados completos (sem PSNR).
    """
    h, w, c = img.shape
    assert c == 3, "Esperada imagem RGB (H, W, 3)."

    original_dtype = img.dtype
    img_float = img.astype(np.float32) / (
        255.0 if img.dtype == np.uint8 else 1.0
    )
    X = img_float.reshape(-1, 3)

    if use_gpu:
        clear_gpu_memory()

    print(f"Comprimindo com K={k}... ", end="", flush=True)
    t0 = time.time()

    try:
        centroids, idx, sse = run_kmeans_single(
            X,
            k,
            max_iters=max_iters,
            seed=seed,
            n_init=n_init,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
    except Exception as exc:
        if use_gpu and "memory" in str(exc).lower():
            print("GPU OOM! Usando CPU... ", end="", flush=True)
            clear_gpu_memory()
            centroids, idx, sse = run_kmeans_single(
                X,
                k,
                max_iters=max_iters,
                seed=seed,
                n_init=n_init,
                use_gpu=False,
                batch_size=batch_size,
            )
        else:
            raise

    elapsed = time.time() - t0
    centroids = centroids.astype(np.float32)

    compressed_img = reconstruct_image(centroids, idx, img.shape, original_dtype)

    unique_orig = count_unique_colors(img)
    unique_comp = count_unique_colors(compressed_img)

    orig_mb = (h * w * 3) / (1024 * 1024)
    idx_dtype = get_optimal_dtype(k)

    comp_bytes = k * 3 * 4 + idx.size * np.dtype(idx_dtype).itemsize
    comp_mb = comp_bytes / (1024 * 1024)

    ratio = orig_mb / comp_mb if comp_mb > 0 else float("inf")

    print(f"✓ {elapsed:.1f}s | {ratio:.2f}x")

    if use_gpu:
        clear_gpu_memory()

    gc.collect()

    return {
        "K": k,
        "tempo_s": round(elapsed, 4),
        "centroids": centroids,
        "idx": idx,
        "compressed_img": compressed_img,
        "SSE": sse,
        "cores_originais": unique_orig,
        "cores_comprimidas": unique_comp,
        "tamanho_original_MB": orig_mb,
        "tamanho_comprimido_MB": comp_mb,
        "fator_compactacao": ratio,
        "idx_dtype": str(idx_dtype.__name__),
    }


def run_full_experiment(
    image_path,
    k,
    max_iters=10,
    seed=0,
    n_init=1,
    use_gpu=True,
    batch_size=200000
):
    """
    Carrega a imagem, roda o K-Means e retorna resultado completo.
    """
    original_img = imageio.imread(image_path)

    result = compress_image(
        original_img,
        k=k,
        max_iters=max_iters,
        seed=seed,
        n_init=n_init,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    file_size_bytes = os.path.getsize(image_path)
    jpeg_size_mb = file_size_bytes / (1024 * 1024)

    result["original_img"] = original_img
    result["jpeg_size_mb"] = jpeg_size_mb

    return result
