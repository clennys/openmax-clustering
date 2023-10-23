import numpy as np
import torch
import matplotlib.pyplot as plt


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


def known_unknown_acc(openmax_predictions_per_model, num_clusters_per_class=1):
    for key in openmax_predictions_per_model:
        print(f" ======= {key} ====== ")
        knowns_acc = [0, 0]
        unknown_acc = [0, 0]
        condensed_cluster_to_class = {}
        # if num_clusters_per_class > 100:
        #     condensed_cluster_to_class = clusters_to_class(
        #         openmax_predictions_per_model[key], num_clusters_per_class
        #     )
        # else:
        condensed_cluster_to_class = openmax_predictions_per_model[key]
        for label in sorted(condensed_cluster_to_class):
            label_tensor = condensed_cluster_to_class[label]
            counts = torch.sum(label_tensor.eq(label))
            total = label_tensor.size(dim=0)
            if label != -1:
                knowns_acc[0] += int(counts.item())
                knowns_acc[1] += total
            else:
                unknown_acc[0] = int(counts.item())
                unknown_acc[1] = total
            print(f"Acc per label {label}: {counts} / {total}")
        print(
            f"\nKnown: {knowns_acc[0]/knowns_acc[1]:.3f}, Unknown {unknown_acc[0]/unknown_acc[1]:.3f} \n"
        )


def ccr_fpr_plot(ccr_fpr_per_model):
    fig, axs = plt.subplots(1, 1)
    for key in ccr_fpr_per_model.keys():
        axs.plot(ccr_fpr_per_model[key][1], ccr_fpr_per_model[key][0], label=key)
    plt.legend(loc="lower right")
    plt.show()
