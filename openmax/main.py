from collections import Counter
import torch
import torch.nn as nn
import sys
from datasets import EMNIST
from train import *
from val import *
from model import LeNet
from openmax_algo import *
from loguru import logger
import matplotlib.pyplot as plt
from metrics import *
import torchvision.models as models
import argparse


def logger_setup(debug_output:bool):
    logger.remove()
    if debug_output:
        logger.add("./logs/debug_log_{time}.log", level="DEBUG")
    logger.add(sys.stderr, level="INFO")

def args_setup():
    parser = argparse.ArgumentParser(prog='CNN with MNIST and OpenMax',
                    description='TODO: runs models',
                    epilog='TODO: May the force be with you.')
    parser.add_argument('--base', '-b',  action='store_true',  help='Runs the baseline model')
    parser.add_argument('--input-cluster', '-ic',  type=int, help='Runs the cluster model with the specified number of clusters')
    parser.add_argument('--feature-cluster', '-fc',  type=int, help='Runs the cluster model with the specified number of clusters')
    parser.add_argument('--debug', '-d', action='store_true', help='activate debug output in log file')
    parser.add_argument('--epochs', '-e',  type=int, help='number of epochs during the training phase')
    parser.add_argument('--train-only', '-to', action='store_true', help='only train network')

    return parser


def baseline_model(epochs, train_only):
    BATCH_SIZE = 32
    EPOCHS = epochs if epochs is not None else 1
    logger.info(f"Using Baseline model with {BATCH_SIZE}")

    training_data = EMNIST(
        dataset_root="./downloads/",
        which_set="train",
        include_unknown=False,
        has_garbage_class=False,
    )
    validation_data = EMNIST(
        dataset_root="./downloads/",
        which_set="val",
        include_unknown=True,
        has_garbage_class=False,
    )
    test_data = EMNIST(
        dataset_root="./downloads/",
        which_set="test",
        include_unknown=False,
        has_garbage_class=False,
    )

    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=BATCH_SIZE, pin_memory=True
    )

    model = LeNet(
        use_classification_layer=True,
        use_BG=False,
        num_classes=10,
        final_layer_bias=True,
    )

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    path_model = "./saved_models/openmax_cnn_eminst0.pth"

    training_features_dict = train(
        model, training_data, train_data_loader, optimizer, loss_fn, EPOCHS, path_model
    )

    if not train_only:
        val_features_dict, val_logits_dict = validation(
            model, val_data_loader, validation_data, loss_fn, path_model
        )

        tail_sizes = [10, 100, 250, 500, 750, 1000]
        logger.info(f"openmax: tail_sizes {tail_sizes}")

        distance_multpls = [1.50, 1.7, 2.0, 2.3]
        logger.info(f"openmax: distance_multpls {distance_multpls}")

        _, _, openmax_predictions_per_model, openmax_scores_per_model = openmax_run(
            tail_sizes,
            distance_multpls,
            training_features_dict,
            val_features_dict,
            val_logits_dict,
            alpha=8,
        )

        known_unknown_acc(openmax_predictions_per_model)

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

        # ccr_fpr_plot(ccr_fpr_per_model)


def cluster_model(num_clusters_per_class, epochs, train_only):
    BATCH_SIZE = 32
    EPOCHS = epochs if epochs is not None else 1
    CLUSTERS_PER_CLASS = num_clusters_per_class
    TOTAL_NUM_CLUSTERS = num_clusters_per_class * 10

    logger.info(f"Using Cluster model with {BATCH_SIZE}")

    cluster_training_dataset = EMNIST(
        dataset_root="./downloads/",
        which_set="train",
        include_unknown=False,
        has_garbage_class=False,
        use_clusters=True,
        num_clusters_per_class=CLUSTERS_PER_CLASS,
    )

    cluster_validation_data = EMNIST(
        dataset_root="./downloads/",
        which_set="val",
        include_unknown=True,
        has_garbage_class=False,
        # use_clusters=True,
        # num_clusters_per_class=CLUSTERS_PER_CLASS,
    )

    cluster_train_data_loader = torch.utils.data.DataLoader(
        cluster_training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )

    cluster_val_data_loader = torch.utils.data.DataLoader(
        cluster_validation_data, batch_size=BATCH_SIZE, pin_memory=True
    )

    cluster_model = LeNet(
        use_classification_layer=True,
        use_BG=False,
        num_classes=TOTAL_NUM_CLUSTERS,
        final_layer_bias=True,
    )

    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        cluster_model.parameters(), lr=learning_rate, momentum=0.9
    )

    path_model = (
        f"./saved_models/openmax_cnn_eminst_cluster-{TOTAL_NUM_CLUSTERS}.pth"
    )

    training_features_dict = train(
        cluster_model,
        cluster_training_dataset,
        cluster_train_data_loader,
        optimizer,
        loss_fn,
        EPOCHS,
        path_model,
    )

    if not train_only:
        val_features_dict, val_logits_dict = validation_cluster(
            cluster_model,
            cluster_val_data_loader,
            cluster_validation_data,
            loss_fn,
            CLUSTERS_PER_CLASS,
            path_model,
        )
        tail_sizes = [10, 100, 250, 500, 750, 1000]
        logger.info(f"openmax: tail_sizes {tail_sizes}")

        distance_multpls = [1.50, 1.7, 2.0, 2.3]
        logger.info(f"openmax: distance_multpls {distance_multpls}")

        _, _, openmax_predictions_per_model, openmax_scores_per_model = openmax_run(
            tail_sizes,
            distance_multpls,
            training_features_dict,
            val_features_dict,
            val_logits_dict,
            alpha=10,
            total_num_clusters=TOTAL_NUM_CLUSTERS,
        )

        known_unknown_acc(openmax_predictions_per_model, CLUSTERS_PER_CLASS)

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

        # ccr_fpr_plot(ccr_fpr_per_model)


if __name__ == "__main__":
    args = args_setup().parse_args()
    logger_setup(args.debug)
    if args.base:
        baseline_model(args.epochs, args.train_only)
    elif args.input_cluster is not None:
        cluster_model(args.input_cluster, args.epochs, args.train_only)