import numpy as np


def find_closest_centroids(X, centroids):
    """Encontra o centróide mais próximo para cada exemplo em X."""
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    
    return idx


def compute_centroids(X, idx, K):
    """Calcula os novos centróides com base nas atribuições atuais."""
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        points_assigned_to_centroid = X[idx == i]
        if len(points_assigned_to_centroid) > 0:
            centroids[i] = (1/len(points_assigned_to_centroid)) * np.sum(points_assigned_to_centroid, axis=0)
    
    return centroids


def kMeans_init_centroids(X, K):
    """Inicializa os centróides selecionando K exemplos aleatórios de X."""
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False, plot_function=None):
    """Executa o algoritmo K-Means."""
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m, dtype=int)

    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        idx = find_closest_centroids(X, centroids)
        
        if plot_progress and plot_function is not None:
            plot_function(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx