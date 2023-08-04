import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import datasets

def openmax_alpha():
    pass

def openmax_training():
    pass

def main():

    digits = datasets.HandwrittenDigits()

    kmeans_cluster = KMeans(init="k-means++", n_clusters=10, n_init=10, random_state=0)

    pca = PCA(2)
    reduced_data = pca.fit_transform(digits.get_features())
    kmeans_cluster.fit(reduced_data)

    centroids = kmeans_cluster.cluster_centers_
    label = kmeans_cluster.fit_predict(reduced_data)
    unique_labels = np.unique(label)

    plt.figure(figsize=(8, 8))
    for i in unique_labels:
        plt.scatter(reduced_data[label == i, 0], reduced_data[label == i, 1], label=i) 
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='k', zorder=10)
    plt.legend()
    plt.savefig('output.pdf')


