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
