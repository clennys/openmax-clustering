import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from loguru import logger


def preprocess_oscr(class_scores_dict: dict):
    all_probs_per_model = []
    gt = []
    for key in class_scores_dict.keys():
        all_probs_per_model.append(class_scores_dict[key])
        gt += [key] * len(class_scores_dict[key])
    return np.array(gt), torch.cat(all_probs_per_model).numpy()


def calculate_oscr(gt, scores, unk_label=-1):
    """Calculates the OSCR values, iterating over the score of the target class of every sample,
    produces a pair (ccr, fpr) for every score.
    Args:
        gt (np.array): Integer array of target class labels.
        scores (np.array): Float array of dim [N_samples, N_classes]
        unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)
    Returns: Two lists first one for ccr, second for fpr.
    """
    # Change the unk_label to calculate for kn_unknown or unk_unknown
    gt = gt.astype(int)
    kn = gt >= 0
    unk = gt == unk_label

    # Get total number of samples of each type
    total_kn = np.sum(kn)
    total_unk = np.sum(unk)

    ccr, fpr = [], []
    # get predicted class for known samples
    pred_class = np.argmax(scores, axis=1)[kn]
    correctly_predicted = pred_class == gt[kn]
    target_score = scores[kn][range(kn.sum()), gt[kn]]

    # get maximum scores for unknown samples
    max_score = np.max(scores, axis=1)[unk]

    # Any max score can be a threshold
    thresholds = np.unique(max_score)

    for tau in thresholds:
        # compute CCR value
        val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
        ccr.append(val)

        val = (max_score >= tau).sum() / total_unk
        fpr.append(val)

    ccr = np.array(ccr)
    fpr = np.array(fpr)
    return ccr, fpr


def clusters_to_class(clusters_dict, num_clusters_per_class):
    condensed_cluster_to_class = {}
    for key in clusters_dict.keys():
        if (key // num_clusters_per_class) in condensed_cluster_to_class and key != -1:
            condensed_cluster_to_class[key // num_clusters_per_class] = torch.cat(
                (
                    condensed_cluster_to_class[key // num_clusters_per_class],
                    clusters_dict[key],
                )
            )
        elif key == -1:
            condensed_cluster_to_class[key] = clusters_dict[key]
        else:
            condensed_cluster_to_class[key // num_clusters_per_class] = clusters_dict[
                key
            ]
    return condensed_cluster_to_class


def known_unknown_acc(
    openmax_predictions_per_model, alpha, num_clusters_per_class=1, max_condense=True
):
    acc_per_model = {}
    if max_condense:
        num_clusters_per_class = 1
    for key in openmax_predictions_per_model:
        logger.debug(f" ======= {key} - {alpha} ====== ")
        knowns_acc = [0, 0]
        unknown_acc = [0, 0]
        condensed_cluster_to_class = {}
        accuracy = {}
        condensed_cluster_to_class = openmax_predictions_per_model[key]
        for label in sorted(condensed_cluster_to_class):
            label_tensor = torch.where(
                condensed_cluster_to_class[label] != -1,
                condensed_cluster_to_class[label].int() // num_clusters_per_class,
                condensed_cluster_to_class[label],
            )
            counts = torch.sum(label_tensor.eq(label))
            total = label_tensor.size(dim=0)
            if label != -1:
                knowns_acc[0] += int(counts.item())
                knowns_acc[1] += total
            else:
                unknown_acc[0] = int(counts.item())
                unknown_acc[1] = total
            logger.debug(f"Acc per label {label}: {counts} / {total}")
            accuracy[label] = (counts, total)
        logger.debug(
            f"\nKnown: {knowns_acc[0]/knowns_acc[1]:.3f}, Unknown {unknown_acc[0]/unknown_acc[1]:.3f} \n"
        )
        acc_per_model[key] = accuracy
    return acc_per_model


def oscr_confidence(gt, scores, unknown_label=-1):
    knowns = gt >= 0
    n_knowns = np.sum(knowns)

    unknowns = gt == unknown_label
    n_unknowns = np.sum(unknowns)

    scores_unknown_samples = scores[unknowns]

    gamma_negative = 1/n_unknowns * np.sum(1 - np.max(scores_unknown_samples, axis=1))

    known_scores = scores[knowns]
    scores_gt =  known_scores[np.arange(known_scores.shape[0]), gt[knowns]]

    gamma_positive = 1/n_knowns * np.sum(scores_gt)

    gamma = (gamma_positive + gamma_negative) * 0.5

    return gamma, gamma_positive, gamma_negative

def oscr_epsilon_ccr_at_fpr(ccr, fpr, fpr_thresholds):
    threshold_values = []
    for threshold in fpr_thresholds:
        idx = np.where(fpr==threshold)
        value = ccr[idx]
        if value.shape[0] == 0: 
            value = np.append(value, 0.0)
        threshold_values.append((threshold, value))
    return np.sum([t[1] for t in threshold_values]), threshold_values


def ccr_fpr_plot(ccr_fpr_per_model):
    fig, axs = plt.subplots(1, 1)
    for key in ccr_fpr_per_model.keys():
        axs.plot(ccr_fpr_per_model[key][1], ccr_fpr_per_model[key][0], label=key)
    plt.legend(loc="lower right")
    plt.show()


def oscr(openmax_scores_per_model):
    processed_oscr_openmax_scores_per_model: dict = {}
    for model_key in openmax_scores_per_model.keys():
        processed_oscr_openmax_scores_per_model[model_key] = preprocess_oscr(
            openmax_scores_per_model[model_key]
        )

    ccr_fpr_per_model: dict = {}
    for model_key in processed_oscr_openmax_scores_per_model.keys():
        ccr_fpr_per_model[model_key] = calculate_oscr(
            processed_oscr_openmax_scores_per_model[model_key][0],
            processed_oscr_openmax_scores_per_model[model_key][1],
        )
    return ccr_fpr_per_model


def save_oscr_values(
    path,
    model_type,
    oscr_dict,
    alpha,
    negative_fix,
    acc_per_model,
    normalize_factor,
    cluster_input=1,
    cluster_feature=1,
):
    file_ = (
        path
        + f"oscr_data_{model_type}_{cluster_input}_{cluster_feature}_{alpha}_{negative_fix}_{normalize_factor}.pkl"
    )
    with open(file_, "wb") as f:
        pickle.dump((oscr_dict, acc_per_model), f)
    logger.info(f"OSCR Data saved as {file_}.")
