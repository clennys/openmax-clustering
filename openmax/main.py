from collections import Counter
import torch
import torch.nn as nn
import sys
from datasets import EMNIST
from train import *
from val import *
from test import *
from model import LeNet
from openmax_algo import *
from loguru import logger
import matplotlib.pyplot as plt
from metrics import *
import torchvision as models
import argparse
import pickle
from util import Params


def logger_setup(debug_output: bool, path_dir):
    logger.remove()
    if debug_output:
        logger.add(path_dir + "debug_log_{time}.log", level="DEBUG")
    logger.add(sys.stderr, level="INFO")


def args_setup():
    parser = argparse.ArgumentParser(
        prog="CNN with MNIST and OpenMax",
        description="TODO: runs models",
        epilog="TODO: May the force be with you.",
    )
    parser.add_argument(
        "--base", "-b", action="store_true", help="Runs the baseline model"
    )
    parser.add_argument(
        "--input-cluster",
        "-ic",
        type=int,
        help="Runs the cluster model with the specified number of clusters",
    )
    parser.add_argument(
        "--feature-cluster",
        "-fc",
        type=int,
        help="Runs the cluster model with the specified number of clusters",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="activate debug output in log file"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, help="number of epochs during the training phase"
    )
    parser.add_argument(
        "--train-only", "-to", action="store_true", help="only train network"
    )

    parser.add_argument('filename')

    return parser

def save_oscr_values(path, model_type, num_cluster_per_class, oscr_dict, alpha, negative_fix):
    file_ = path + f"oscr_data_{model_type}_{num_cluster_per_class}_{alpha}_{negative_fix}.pkl"
    with open(file_, 'wb') as f:
        pickle.dump(oscr_dict, f)
    logger.info(f"OSCR Data saved as {file_}.")
        

def baseline_model(params):
    BATCH_SIZE = params.batch_size
    EPOCHS =  params.epochs
    logger.info(f"Using Baseline model with Batch-size = {BATCH_SIZE}")

    training_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="train",
        include_unknown=False,
        has_garbage_class=False,
    )
    validation_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="val",
        include_unknown=True,
        has_garbage_class=False,
    )
    test_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="test",
        include_unknown=True,
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

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, pin_memory=True
    )

    model = LeNet(
        use_classification_layer=True,
        use_BG=False,
        num_classes=10,
        final_layer_bias=True,
    )

    learning_rate = params.learning_rate
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    path_model = params.saved_models_dir + "openmax_cnn_eminst0.pth"

    training_features_dict = train(
        model, training_data, train_data_loader, optimizer, loss_fn, EPOCHS, path_model
    )

    if not params.train_only:
        val_features_dict, val_logits_dict = validation(
            model, val_data_loader, validation_data, loss_fn, path_model
        )

        test_features_dict, test_logits_dict = test(
            model, test_data_loader, test_data, loss_fn, path_model
        )


        tail_sizes = params.tail_sizes
        logger.info(f"openmax: tail_sizes {tail_sizes}")

        distance_multpls = params.distance_multpls
        logger.info(f"openmax: distance_multpls {distance_multpls}")


        for alpha in params.alphas:
            for negative_fix in params.negative_fix:
                logger.info(f"Negative shift: {negative_fix}")
                _, _, openmax_predictions_per_model, openmax_scores_per_model = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    training_features_dict,
                    test_features_dict,
                    test_logits_dict,
                    alpha=alpha,
                    negative_fix=negative_fix
                )

                known_unknown_acc(openmax_predictions_per_model, alpha)

                ccr_fpr_per_model = oscr(openmax_scores_per_model)

                save_oscr_values(params.experiment_data_dir, "base", 1, ccr_fpr_per_model, alpha, params.negative_fix)


def input_cluster_model(params):
    BATCH_SIZE = params.batch_size
    EPOCHS = params.epochs
    for n_clusters in params.num_clusters_per_class:
        CLUSTERS_PER_CLASS = n_clusters
        TOTAL_NUM_CLUSTERS = CLUSTERS_PER_CLASS * 10

        logger.info(f"Using Cluster model with {BATCH_SIZE}")

        cluster_training_dataset = EMNIST(
            dataset_root=params.emnist_dir,
            which_set="train",
            include_unknown=False,
            has_garbage_class=False,
            use_clusters=True,
            num_clusters_per_class=CLUSTERS_PER_CLASS,
        )

        cluster_validation_data = EMNIST(
            dataset_root=params.emnist_dir,
            which_set="val",
            include_unknown=True,
            has_garbage_class=False,
        )

        cluster_test_data = EMNIST(
            dataset_root=params.emnist_dir,
            which_set="test",
            include_unknown=True,
            has_garbage_class=False,
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

        cluster_test_data_loader = torch.utils.data.DataLoader(
            cluster_test_data, batch_size=BATCH_SIZE, pin_memory=True
        )

        cluster_model = LeNet(
            use_classification_layer=True,
            use_BG=False,
            num_classes=TOTAL_NUM_CLUSTERS,
            final_layer_bias=True,
        )

        learning_rate = params.learning_rate,
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            cluster_model.parameters(), lr=learning_rate, momentum=0.9
        )

        path_model = params.saved_models_dir + f"openmax_cnn_eminst_cluster-{TOTAL_NUM_CLUSTERS}.pth"

        training_features_dict = train(
            cluster_model,
            cluster_training_dataset,
            cluster_train_data_loader,
            optimizer,
            loss_fn,
            EPOCHS,
            path_model,
            input_clustering=True
        )

        if not params.train_only:
            val_features_dict, val_logits_dict = validation_input_cluster(
                cluster_model,
                cluster_val_data_loader,
                cluster_validation_data,
                loss_fn,
                CLUSTERS_PER_CLASS,
                path_model,
            )

            test_features_dict, test_logits_dict = test_input_cluster(
                cluster_model,
                cluster_test_data_loader,
                cluster_test_data,
                loss_fn,
                CLUSTERS_PER_CLASS,
                path_model,
            )

            tail_sizes = params.tail_sizes
            logger.info(f"openmax: tail_sizes {tail_sizes}")

            distance_multpls = params.distances_multpls
            logger.info(f"openmax: distance_multpls {distance_multpls}")

            negative_fix = None

            for alpha in params.alphas:

                _, _, openmax_predictions_per_model, openmax_scores_per_model = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    training_features_dict,
                    test_features_dict,
                    test_logits_dict,
                    alpha=alpha,
                    negative_fix = negative_fix,
                    n_clusters_per_class=CLUSTERS_PER_CLASS,
                )

                known_unknown_acc(openmax_predictions_per_model, CLUSTERS_PER_CLASS)

                ccr_fpr_per_model = oscr(openmax_scores_per_model)
                
                save_oscr_values(params.eperiment_data_dir,"base", n_clusters, ccr_fpr_per_model, alpha, params.negative_fix)

def feature_cluster_model(params):
    BATCH_SIZE = params.batch_size
    EPOCHS = params.epochs

    logger.info(f"Using Feature cluster model with batch-size {BATCH_SIZE}")

    training_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="train",
        include_unknown=False,
        has_garbage_class=False,
    )
    validation_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="val",
        include_unknown=False,
        has_garbage_class=False,
    )
    test_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="test",
        include_unknown=True,
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

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, pin_memory=True
    )

    model = LeNet(
        use_classification_layer=True,
        use_BG=False,
        num_classes=10,
        final_layer_bias=True,
    )

    learning_rate = params.learning_rate
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    path_model = params.saved_models_dir + "openmax_cnn_eminst0.pth"

    training_features_dict = train(
        model, training_data, train_data_loader, optimizer, loss_fn, EPOCHS, path_model
    )

    if not params.train_only:
        val_features_dict, val_logits_dict = validation(
            model, val_data_loader, validation_data, loss_fn, path_model
        )

        tail_sizes = params.tail_sizes
        logger.info(f"openmax: tail_sizes {tail_sizes}")

        distance_multpls = params.distance_multpls
        logger.info(f"openmax: distance_multpls {distance_multpls}")

        test_features_dict, test_logits_dict = test_feature_cluster(
            model, test_data_loader, test_data, loss_fn, path_model
        )

        negative_fix = None

        for n_clusters in params.num_clusters_per_class:
            CLUSTERS_PER_CLASS = n_clusters

            for alpha in params.alphas:

                _, _, openmax_predictions_per_model, openmax_scores_per_model = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    val_features_dict,
                    test_features_dict,
                    test_logits_dict,
                    negative_fix=negative_fix,
                    alpha=alpha,
                    n_clusters_per_class=CLUSTERS_PER_CLASS,
                    feature_cluster=True
                )

                known_unknown_acc(openmax_predictions_per_model, alpha, CLUSTERS_PER_CLASS)

                ccr_fpr_per_model = oscr(openmax_scores_per_model)

                # Save Data as npz
                save_oscr_values(params.experiment_data_dir, "input-cluster", CLUSTERS_PER_CLASS, ccr_fpr_per_model, alpha, negative_fix)


if __name__ == "__main__":
    args = args_setup().parse_args()
    params = Params(args.filename)
    logger_setup(params.logger_output, params.log_dir)
    if params.type == "base":
        baseline_model(params)
    elif params.type == "input-cluster":
        input_cluster_model(params)
    elif params.type == "feature-cluster":
        feature_cluster_model(params)
