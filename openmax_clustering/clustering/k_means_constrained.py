from k_means_constrained import KMeansConstrained


def kmeans_constrained(n_clusters_, size_min_, size_max_, clustering_data):
    clusterer_kmeans_c = KMeansConstrained(
        n_clusters=n_clusters_, size_min=size_min_, size_max=size_max_
    )
    clusterer_kmeans_c.fit(clustering_data)
    return clusterer_kmeans_c
