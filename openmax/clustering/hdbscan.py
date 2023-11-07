import hdbscan as hdb


def hdbscan(min_cluster_size_, min_samples_, clustering_data):
    clusterer_hdbscan = hdb.HDBSCAN(
        min_cluster_size=min_cluster_size_, min_samples=min_samples_
    ).fit(clustering_data)

    n_clusters_ = len(set(clusterer_hdbscan.labels_)) - (
        1 if -1 in clusterer_hdbscan.labels_ else 0
    )
    n_noise_ = list(clusterer_hdbscan.labels_).count(-1)

    return clusterer_hdbscan, n_clusters_, n_noise_
