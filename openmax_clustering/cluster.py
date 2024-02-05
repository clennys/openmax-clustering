from util.util import load_network_output
from openset.openmax import features_clustering
import pickle
import torch

dataset_name = "emnist"
val_clustering = False
training_clustering = True
input_clustering = True
feature_clustering = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def cluster_output_to_pkl(data, path, model_name, n_clusters):
    type_c = "training" if training_clustering else "validation"
    file_ = path + "dnn_cluster_" + f"{model_name}_{type_c}_{n_clusters}" + ".pkl"
    with open(file_, "wb") as f:
        pickle.dump(data, f)
        print(f"Saved:{file_}")


def condensed(data, n_clusters):
    condensed_dict = {}
    for key, tensor in data.items():
        new_key = key // n_clusters
        if new_key in condensed_dict:
            condensed_dict[new_key] = torch.vstack((condensed_dict[new_key], tensor))
        else:
            condensed_dict[new_key] = tensor
    return condensed_dict


for f in feature_clustering:
    model_name = f"openmax_cnn_{dataset_name}_cluster-{10*f}"
    path = "./saved_models/network_outputs/"
    path_cluster = "./saved_models/clusters/"

    (
        training_features_dict,
        val_features_dict,
        val_logits_dict,
        test_features_dict,
        test_logits_dict,
    ) = load_network_output(path, model_name)

    data = None
    if val_clustering:
        data = val_features_dict
    elif training_clustering and input_clustering:
        data = condensed(training_features_dict, f)
        for key in data:
            print(key, data[key].shape)
    elif training_clustering:
        data = training_features_dict

    for n_clusters in feature_clustering:
        cluster_dict = features_clustering(data, n_clusters)
        cluster_output_to_pkl(cluster_dict, path_cluster, model_name, n_clusters)
        cluster_dict = None
