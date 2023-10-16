import torch
from vast.opensetAlgos.openmax import fit_high
from torch import Tensor
import torch.nn as nn


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
    weights[mask] = 1 + (
        1 - weights[mask]
    )  # Change negative values to 50.0 or any other desired value
    return sorted_activations * weights

def openmax_training(
    features_labels_all_classes_dict: dict,
    distance_multiplier: float = 2.,
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

        distances: Tensor = cosine_pairwisedistance(
            features, mean_av_tensor[None, :]
        )
        model_per_label["weibull"] = fit_high(
            distances.T, distance_multiplier, tailsize, translateAmount=1
        )
        model_[label] = model_per_label

    return model_

def openmax_inference(features_all_classes: dict, model_) -> dict:  # dict[str, Tensor],
    # features: Tensor = torch.cat(list(features_all_classes.values()))
    props_dict: dict = {}
    for class_label in features_all_classes:
        features = features_all_classes[class_label]
        probs = []
        for model_label in sorted(model_.keys()):
            # mav: Tensor = torch.stack(list(mean_av_dict.values()))
            mav: Tensor = model_[model_label]["mav"]
            distances: Tensor = cosine_pairwisedistance(features, mav)
            probs.append(
                model_[model_label]["weibull"].wscore(distances, isReversed=True)
            )  # TODO: Reversed? 1 - weibull.cdf
        probs = torch.cat(probs, dim=1)
        props_dict[class_label] = probs

    return props_dict

def openmax_alpha(
    evt_probs: Tensor, activations: Tensor, alpha: int, negative_fix=None, debug=False, ignore_unknown_class=False
):
    # Convert to Unknowness
    per_class_unknownness_prob = 1 - evt_probs

    # Line 1: Sort for highest activation value
    sorted_activations, indices = torch.sort(activations, descending=True, dim=1)
    if debug:
        print(activations.shape)
        print(sorted_activations.shape)
        print(indices.shape)

    # Create weights of ones in correct shape
    weights = torch.ones(activations.shape[0], activations.shape[1])
    if debug:
        print("LINE [1]: weights", weights)

    # Line 2-4
    # Creating a sequence of integers from 1 to alpha (inclusive) with a stepsize 1
    # Sequence is assigned to the first alpha columns of all rows in weights
    weights[:, :alpha] = torch.arange(1, alpha + 1, step=1)
    if debug:
        print("LINE [2-4]a: weights", weights)

    # Subtracts the current value in these position alhpa and then divides the result by alpha (elementwise)
    weights[:, :alpha] = (alpha - weights[:, :alpha]) / alpha
    if debug:
        print("LINE [2-4]b: weights", weights)

    weights[:, :alpha] = 1 - weights[:, :alpha] * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )
    if debug:
        print("LINE [2-4]c: weights", weights)

    if negative_fix == "VALUE_SHIFT":
        assert not torch.any(
            torch.lt(weights, 0)
        ), "The tensor contains a negative value"
        # Get the minimum values for each row
        min_values = torch.min(sorted_activations, dim=1).values

        # Reshape the min_values tensor to match the shape of 'a' for broadcasting
        min_values_reshaped = min_values.view(-1, 1)
        min_values_reshaped_abs = torch.abs(min_values_reshaped)

        # Add the minimum values to the corresponding rows of the original tensor
        sorted_activations = torch.add(sorted_activations, min_values_reshaped_abs)

        if debug:
            print("LINE [2-4]d VALUE_SHIFT:", sorted_activations)

    # Line 5
    if negative_fix == "ABS_REV_ACTV":
        revisted_activations = multiply_tensors_with_sign(sorted_activations, weights)
        if debug:
            print("LINE [5] ABS_REV_ACTV:", revisted_activations)
    else:
        revisted_activations = sorted_activations * weights
        if debug:
            print("LINE [5]: ", revisted_activations)

    # Line 6
    unknowness_class_prob = torch.sum(sorted_activations * (1 - weights), dim=1)
    revisted_activations = torch.scatter(
        torch.ones(revisted_activations.shape), 1, indices, revisted_activations
    )
    if debug:
        print("LINE [6]a: unknowness class p", unknowness_class_prob)
        print("LINE [6]b: revisted_act", revisted_activations)

    probability_vector = torch.cat(
        [unknowness_class_prob[:, None], revisted_activations], dim=1
    )
    if debug:
        print("LINE [6]c: prob vec", probability_vector)

    # Line 7
    probability_vector = torch.nn.functional.softmax(probability_vector, dim=1)
    if debug:
        print("LINE [7]: softmax prob vec", probability_vector)
        
    if ignore_unknown_class:
        probs_kkc = probability_vector[:, 1:].clone().detach()
        assert probs_kkc.shape == activations.shape
        return probs_kkc

    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    if debug:
        print("LINE [8]a: prediction_score", prediction_score)
    if debug:
        print("LINE [8]b: predicted_class", predicted_class)

    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score
