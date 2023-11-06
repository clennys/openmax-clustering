from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from k_means_constrained import KMeansConstrained
import hdbscan as hdb


def agglo_clustering(n_clusters_, linkage_, metric_, clustering_data):
    clusterer_agglo = AgglomerativeClustering(
        n_clusters=n_clusters_, linkage=linkage_, metric=metric_
    )
    clusterer_agglo.fit_predict(clustering_data)
    return clusterer_agglo


def hdbscan(min_cluster_size_, min_samples_, clustering_data):
    clusterer_hdbscan = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size_, min_samples=min_samples_
    ).fit(clustering_data)

    n_clusters_ = len(set(clusterer_hdbscan.labels_)) - (
        1 if -1 in clusterer_hdbscan.labels_ else 0
    )
    n_noise_ = list(clusterer_hdbscan.labels_).count(-1)

    return clusterer_hdbscan, n_clusters_, n_noise_


def dbscan(eps_: int, min_samples_: int, clustering_data):
    clusterer_dbscan = DBSCAN(eps=eps_, min_samples=min_samples_, metric="cosine").fit(
        clustering_data
    )

    n_clusters_ = len(set(clusterer_dbscan.labels_)) - (
        1 if -1 in clusterer_dbscan.labels_ else 0
    )
    n_noise_ = list(clusterer_dbscan.labels_).count(-1)

    return clusterer_dbscan, n_clusters_, n_noise_


def kmeans_constrained(n_clusters_, size_min_, size_max_, clustering_data):
    clusterer_kmeans_c = KMeansConstrained(
        n_clusters=n_clusters_, size_min=size_min_, size_max=size_max_
    )
    clusterer_kmeans_c.fit(clustering_data)
    return clusterer_kmeans_c
