import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def update(dot, labels, k):
    centroids = np.zeros((k, dot.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(dot[labels == i], axis=0)
    return centroids


def kmeans(dot, k, e=0.001):
    centroids = dot[np.random.choice(dot.shape[0], k, replace=False)]
    distances = np.zeros((dot.shape[0], centroids.shape[0]))

    while True:
        for i in range(centroids.shape[0]):
            distances[:, i] = np.linalg.norm(dot - centroids[i], axis=1)
        cluster = np.argmin(distances, axis=1)
        new_centroids = update(dot, cluster, k)

        if np.linalg.norm(new_centroids - centroids) < e:
            break
        centroids = new_centroids

    return centroids, cluster


def plot_kmeans(dot, k, centroids, labels):
    plt.figure(figsize=(8, 6))
    for i in range(k):
        points = dot[labels == i]
        plt.scatter(points[:, 0], points[:, 1], s=20, label='Cluster {}'.format(i + 1))

        if len(points) > 2:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, label='Centroid')
    plt.title('K-means')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dots = np.random.rand(250, 2)
    k = 3

    centroids, cluster = kmeans(dots, k)

    plot_kmeans(dots, k, centroids, cluster)
