from collections import Counter
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import openmax_clustering.clustering.agglomerative_clustering as cl
from loguru import logger


def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2, 1)


class EMNIST(torch.utils.data.dataset.Dataset):
    """A split dataset for our experiments. It uses MNIST as known samples and EMNIST letters as unknowns.
    Particularly, the 11 letters will be used as negatives (for training and validation), and the 11 letters will serve as unknowns (for testing only) -- we removed letters `g`, `l`, `i` and `o` due to large overlap to the digits.
    The MNIST test set is used both in the validation and test split of this dataset.

    For the test set, you should consider to leave the parameters `include_unknown` and `has_garbage_class` at their respective defaults -- this might make things easier.

    Parameters:

    dataset_root: Where to find/download the data to.

    which_set: Which split of the dataset to use; can be 'train' , 'test' or 'validation' (anything besides 'train' and 'test' will be the validation set)

    include_unknown: Include unknown samples at all (might not be required in some cases, such as training with plain softmax)

    has_garbage_class: Set this to True when training softmax with background class. This way, unknown samples will get class label 10. If False (the default), unknown samples will get label -1.
    """

    def __init__(
        self,
        dataset_root,
        which_set="train",
        include_unknown=True,
        has_garbage_class=False,
        use_clusters=False,
        num_clusters_per_class=1,
    ):
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train" or which_set == "val",
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose]),
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split="letters",
            transform=transforms.Compose([transforms.ToTensor(), transpose]),
        )
        self.which_set = which_set
        targets = (
            list()
            if not include_unknown
            else [1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14]
            if which_set != "test"
            else [
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
            ]
        )
        self.letter_indexes = [
            i for i, t in enumerate(self.letters.targets) if t in targets
        ]
        self.has_garbage_class = has_garbage_class
        self.use_clusters = use_clusters
        self.num_clusters_per_class = num_clusters_per_class

        if use_clusters:
            self.clustering()

    def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index]
        else:
            return (
                self.letters[self.letter_indexes[index - len(self.mnist)]][0],
                10 if self.has_garbage_class else -1,
            )

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes)

    def clustering(self):
        logger.info("Starting Clustering...")
        cluster_data = self.mnist.data.numpy()
        cluster_targets = self.mnist.targets.numpy()
        cluster_data_reshaped = cluster_data.reshape(cluster_data.shape[0], -1)
        cluster_data_dict = {}
        for i in range(len(cluster_data)):
            label = cluster_targets[i]
            if label in cluster_data_dict.keys():
                cluster_data_dict[label] = np.vstack(
                    [cluster_data_dict[label], cluster_data_reshaped[i]]
                )
            else:
                cluster_data_dict[label] = cluster_data_reshaped[i]

        n_clusters = self.num_clusters_per_class
        linkage = "complete"
        metric = "cosine"
        clusterer_dict = {}
        for key in cluster_data_dict.keys():
            clusterer = cl.agglo_clustering(
                n_clusters, linkage, metric, cluster_data_dict[key]
            )
            # size = len(cluster_data_dict[key])/n_clusters
            # clusterer = cl.kmeans_constrained(n_clusters, np.floor(size).astype('int'), np.ceil(size).astype('int'), cluster_data_dict[key])
            clusterer_dict[key] = clusterer

        targets_ = np.array([], dtype=np.uint8)

        for key in sorted(clusterer_dict.keys()):
            cluster_labels = clusterer_dict[key].labels_
            cluster_labels_new = cluster_labels + key * n_clusters
            targets_ = np.append(targets_, cluster_labels_new)

        reshaped_matrices = []
        for key in sorted(clusterer_dict.keys()):
            img_reshaped = cluster_data_dict[key].reshape(6000, 28, 28)
            reshaped_matrices.append(img_reshaped)
        data_ = np.array(np.concatenate(reshaped_matrices))
        logger.info(Counter(targets_))

        self.mnist.data = torch.tensor(data_)
        self.mnist.targets = targets_
        logger.info("Clustering Done.")
