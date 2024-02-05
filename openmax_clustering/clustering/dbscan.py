from sklearn.cluster import DBSCAN


def dbscan(eps_: int, min_samples_: int, clustering_data):
    clusterer_dbscan = DBSCAN(eps=eps_, min_samples=min_samples_, metric="cosine").fit(
        clustering_data
    )

    n_clusters_ = len(set(clusterer_dbscan.labels_)) - (
        1 if -1 in clusterer_dbscan.labels_ else 0
    )
    n_noise_ = list(clusterer_dbscan.labels_).count(-1)

    return clusterer_dbscan, n_clusters_, n_noise_
