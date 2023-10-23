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


def init_logger():
    logger.remove()
    logger.add("./logs/debug_log_{time}.log", level="DEBUG")
    logger.add(sys.stderr, level="INFO")


def baseline_model():
    BATCH_SIZE = 32
    EPOCHS = 1
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

    val_features_dict, val_logits_dict = validation(
        model, val_data_loader, validation_data, loss_fn, path_model
    )

    tail_sizes = [10, 100, 250, 500, 750, 1000]
    logger.info(f"openmax: tail_sizes {tail_sizes}")

    distance_multpls = [1.50, 1.7, 2.0, 2.3]
    logger.info(f"openmax: distance_multpls {distance_multpls}")

    _, _, openmax_predictions_per_model,openmax_scores_per_model = openmax_run(
        tail_sizes,
        distance_multpls,
        training_features_dict,
        val_features_dict,
        val_logits_dict,
        alpha=8
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

 
def cluster_model():
    BATCH_SIZE = 32
    EPOCHS = 1
    CLUSTERS_PER_CLASS = 3

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
        use_clusters=True,
        num_clusters_per_class=CLUSTERS_PER_CLASS,
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
        num_classes=10 * CLUSTERS_PER_CLASS,
        final_layer_bias=True,
        use_cluster=False,
        num_clusters=CLUSTERS_PER_CLASS
    )

    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        cluster_model.parameters(), lr=learning_rate, momentum=0.9
    )

    path_model = f"./saved_models/openmax_cnn_eminst_cluster-{CLUSTERS_PER_CLASS*10}.pth"

    training_features_dict = train(
        cluster_model,
        cluster_training_dataset,
        cluster_train_data_loader,
        optimizer,
        loss_fn,
        EPOCHS,
        path_model,
    )

    # for label in range(10*CLUSTERS_PER_CLASS):
    #     if label not in training_features_dict:
    #         # training_features_dict[label] = torch.full((1, 500), float('inf')) 
    #         training_features_dict[label] = torch.full((1, 500), 0.) 


    val_features_dict, val_logits_dict = validation_cluster(
        cluster_model, cluster_val_data_loader, cluster_validation_data, loss_fn, CLUSTERS_PER_CLASS, path_model
    )
    tail_sizes = [10, 100, 250, 500, 750, 1000]
    logger.info(f"openmax: tail_sizes {tail_sizes}")

    distance_multpls = [1.50, 1.7, 2.0, 2.3]
    logger.info(f"openmax: distance_multpls {distance_multpls}")

    _, _, openmax_predictions_per_model,openmax_scores_per_model = openmax_run(
        tail_sizes,
        distance_multpls,
        training_features_dict,
        val_features_dict,
        val_logits_dict,
        alpha=10,
        cluster_per_class=CLUSTERS_PER_CLASS,
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
    init_logger()
    cluster_model()
    # baseline_model()
