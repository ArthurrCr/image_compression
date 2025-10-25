import matplotlib.pyplot as plt


def plot_kMeans_RGB(X, centroids, idx, K):
    """Plota o resultado do K-Means no espaço de cores RGB (3D)."""
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'grey', 'olive', 'cyan', 'magenta', 'lime']
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k in range(K):
        cluster_points = X[idx == k]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=colors[k % len(colors)], label=f'Cluster {k+1}', s=10)
    
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, linewidths=3, label='Centroids')
    
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    
    plt.show()


def show_centroid_colors(centroids):
    """Mostra as cores representadas pelos centróides do K-Means."""
    num_centroids = centroids.shape[0]
    
    fig, ax = plt.subplots(1, num_centroids, figsize=(num_centroids * 2, 2))
    
    for i in range(num_centroids):
        ax[i].imshow([[centroids[i]]])
        ax[i].axis('off')
        ax[i].set_title(f'Idx {i}', fontsize=8)
    
    plt.show()