from sklearn.cluster import AgglomerativeClustering


def agglo_clustering(n_clusters_, linkage_, metric_, clustering_data):
    clusterer_agglo = AgglomerativeClustering(
        n_clusters=n_clusters_, linkage=linkage_, metric=metric_
    )
    clusterer_agglo.fit_predict(clustering_data)
    return clusterer_agglo
