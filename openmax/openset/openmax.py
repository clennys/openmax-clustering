import torch
from vast.opensetAlgos.openmax import fit_high
from torch import Tensor
import torch.nn as nn
from clustering.agglomerative_clustering import agglo_clustering


def euclidean_pairwisedistance(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes batched the p-norm distance between each pair of the two collections of row vectors.
    :param x: Tensor of size BxPxM
    :param y: Tensor of size BxRxM
    :returns: A Tensor of shape BxPxR
    """
    return torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")


def cosine_pairwisedistance(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum("nc,ck->nk", [x, y.T])
    distances = 1 - similarity
    return distances


def multiply_tensors_with_sign(sorted_activations, weights):
    mask = sorted_activations < 0
    weights[mask] = 1 + (1 - weights[mask])
    return sorted_activations * weights


def val_features_clustering(features, num_cluster_pro_class):
    cluster_features_dict = {}
    for key in features.keys():
        clusterer = agglo_clustering(
            num_cluster_pro_class, "ward", "euclidean", features[key].numpy()
        )
        for cluster_label, values in zip(clusterer.labels_, features[key]):
            dict_key_cluster = key * num_cluster_pro_class + cluster_label
            if dict_key_cluster in cluster_features_dict:
                cluster_features_dict[dict_key_cluster] = torch.cat(
                    (cluster_features_dict[dict_key_cluster], values[None, :])
                )
            else:
                cluster_features_dict[dict_key_cluster] = values[None, :]
    return cluster_features_dict


def max_props_feature_clustering(props_dict, n_clusters_per_class):
    cluster_max_props_dict = {}
    for key in props_dict.keys():
        props_reshaped = props_dict[key].view(
            props_dict[key].shape[0], 10, n_clusters_per_class
        )
        cluster_max_props_dict[key], _ = torch.max(props_reshaped, dim=2)
    return cluster_max_props_dict


def openmax_run(
    tail_sizes: list,
    distance_multpls: list,
    training_features_dict: dict,
    val_features_dict: dict,
    val_logits_dict: dict,
    alpha: int,
    negative_fix,
    n_clusters_per_class=1,
    feature_cluster=False,
):
    if feature_cluster:
        training_features_dict = val_features_clustering(
            training_features_dict, n_clusters_per_class
        )

    models_dict = {}
    for tail in tail_sizes:
        for dist_mult in distance_multpls:
            model_ = openmax_training(training_features_dict, dist_mult, tail)
            key = f"{tail}-{dist_mult}"
            models_dict[key] = model_

    models_props_dict = {}
    for key in models_dict.keys():
        props_dict: dict = openmax_inference(
            val_features_dict, models_dict[key], n_clusters_per_class * 10
        )

        if feature_cluster:
            props_dict = max_props_feature_clustering(props_dict, n_clusters_per_class)
        models_props_dict[key] = props_dict


    openmax_models_scores = {}
    openmax_models_predictions = {}

    for model_idx in models_dict.keys():
        openmax_scores_dict = {}
        openmax_predictions_dict = {}
        for idx, key in enumerate(models_props_dict[model_idx].keys()):
            assert key == list(val_logits_dict.keys())[idx]
            assert (
                models_props_dict[model_idx][key].shape[1]
                == val_logits_dict[key].shape[1]
            ), f"Shape model props {models_props_dict[model_idx][key].shape[1]}, logits shape {val_logits_dict[key].shape[1]}"

            openmax_predictions_dict[key], _, openmax_scores_dict[key] = openmax_alpha(
                models_props_dict[model_idx][key],
                val_logits_dict[key],
                alpha=alpha,
                negative_fix=negative_fix,
                ignore_unknown_class=False,
            )

        openmax_models_scores[model_idx] = openmax_scores_dict
        openmax_models_predictions[model_idx] = openmax_predictions_dict

    return (
        models_dict,
        models_props_dict,
        openmax_models_predictions,
        openmax_models_scores,
    )


def openmax_training(
    features_labels_all_classes_dict: dict,
    distance_multiplier: float = 2.0,
    tailsize: int = 100,
):
    # Tuple[dict[str, weibull], dict[str, Tensor]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process.
    :param features_all_classes: features of all classes
    """

    model_ = {}
    for label in features_labels_all_classes_dict.keys():
        model_per_label = {}
        features: Tensor = features_labels_all_classes_dict[label]
        mean_av_tensor = torch.mean(features, dim=0)

        model_per_label["mav"] = mean_av_tensor[None, :]

        distances: Tensor = cosine_pairwisedistance(features, mean_av_tensor[None, :])
        model_per_label["weibull"] = fit_high(
            distances.T, distance_multiplier, tailsize, translateAmount=1
        )
        model_[label] = model_per_label

    return model_


def openmax_inference(
    features_all_classes: dict, model_, total_num_clusters
) -> dict:  # dict[str, Tensor],
    props_dict: dict = {}
    for (
        class_label
    ) in features_all_classes:  # TODO: All classes with or without clusters?
        features = features_all_classes[class_label]
        probs = []
        # for model_label in sorted(model_.keys()):
        for model_label in range(total_num_clusters):
            if model_label in model_:
                mav: Tensor = model_[model_label]["mav"]
                distances: Tensor = cosine_pairwisedistance(features, mav)
                probs.append(
                    model_[model_label]["weibull"].wscore(distances, isReversed=True)
                )  # TODO: Reversed? 1 - weibull.cdf
            elif class_label == -1:
                probs.append(torch.zeros(8800, 1))
            else:
                probs.append(torch.zeros(1000, 1))
        probs = torch.cat(probs, dim=1)
        props_dict[class_label] = probs

    return props_dict


def openmax_alpha(
    evt_probs: Tensor,
    activations: Tensor,
    alpha: int,
    negative_fix=None,
    ignore_unknown_class=False,
):
    # Convert to Unknowness
    per_class_unknownness_prob = 1 - evt_probs

    # Line 1: Sort for highest activation value
    sorted_activations, indices = torch.sort(activations, descending=True, dim=1)

    # Create weights of ones in correct shape
    weights = torch.ones(activations.shape[0], activations.shape[1])

    # Line 2-4
    # Creating a sequence of integers from 1 to alpha (inclusive) with a stepsize 1
    # Sequence is assigned to the first alpha columns of all rows in weights
    weights[:, :alpha] = torch.arange(1, alpha + 1, step=1)

    # Subtracts the current value in these position alhpa and then divides the result by alpha (elementwise)
    weights[:, :alpha] = (alpha - weights[:, :alpha]) / alpha

    weights[:, :alpha] = 1 - weights[:, :alpha] * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )

    if negative_fix == "VALUE_SHIFT":
        assert not torch.any(
            torch.lt(weights, 0)
        ), "The tensor contains a negative value"
        assert not torch.any(
            torch.gt(weights, 1)
        ), "The tensor contains value bigger than 1"
        # TODO: What If samples has only positive values
        # Get the minimum values for each row
        min_values = torch.min(sorted_activations, dim=1).values

        # Reshape the min_values tensor to match the shape of 'a' for broadcasting
        min_values_reshaped = min_values.view(-1, 1)
        min_values_reshaped_abs = torch.abs(min_values_reshaped)

        # Add the minimum values to the corresponding rows of the original tensor
        sorted_activations = torch.add(sorted_activations, min_values_reshaped_abs)

    # Line 5
    if negative_fix == "ABS_REV_ACTV":
        assert not torch.any(
            torch.gt(weights, 1)
        ), "The tensor contains value bigger than 1"
        revisted_activations = multiply_tensors_with_sign(sorted_activations, weights)
    else:
        revisted_activations = sorted_activations * weights

    # Line 6
    unknowness_class_prob = torch.sum(sorted_activations * (1 - weights), dim=1)
    revisted_activations = torch.scatter(
        torch.ones(revisted_activations.shape), 1, indices, revisted_activations
    )

    probability_vector = torch.cat(
        [unknowness_class_prob[:, None], revisted_activations], dim=1
    )

    # Line 7
    probability_vector = torch.nn.functional.softmax(probability_vector, dim=1)

    if ignore_unknown_class:
        probs_kkc = probability_vector[:, 1:].clone().detach()
        assert probs_kkc.shape == activations.shape
        return probs_kkc

    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)

    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score, probability_vector[:, 1:].clone().detach()
