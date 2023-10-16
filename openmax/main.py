import torch
import torch.nn as nn

from datasets import EMNIST
from train import train
from val import validation
from model import LeNet
from openmax_algo import *
from loguru import logger


def baseline_model():
    BATCH_SIZE = 32
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
        training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=BATCH_SIZE, pin_memory=True
    )

    model = LeNet(
        use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True
    )

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    path_model = "openmax_cnn_eminst0.pth"

    training_features_dict = train(model, training_data, train_data_loader, optimizer, loss_fn, 2, path_model )

    val_features_dict, val_logits_dict = validation(model, val_data_loader, validation_data, loss_fn, path_model)

    tail_sizes = [10, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    tail_sizes = [750]
    print(tail_sizes)
    distance_multpls = [1.50, 1.7, 2.0, 2.3, 2.5, 2.7, 3., 3.5, 4]
    distance_multpls = [2.0]
    print(distance_multpls)

    models_dict = {}

    for tail in tail_sizes:
        for dist_mult in distance_multpls:
            model_ = openmax_training(training_features_dict, dist_mult, tail)
            key = f"{tail}-{dist_mult}"
            models_dict[key] = model_

    models_props_dict = {}
    for key in models_dict.keys():
        props_dict: dict = openmax_inference(val_features_dict, models_dict[key])
        models_props_dict[key] = props_dict


    openmax_models_predictions = {}

    for model_idx in models_dict.keys():
        openmax_prob_dict = {}
        for idx, key in enumerate(models_props_dict[model_idx].keys()):
            assert key == list(val_logits_dict.keys())[idx]
            assert (
                models_props_dict[model_idx][key].shape[1] == val_logits_dict[key].shape[1]
            )
            
            openmax_prob_dict[key]= openmax_alpha(
                models_props_dict[model_idx][key],
                val_logits_dict[key],
                alpha=9,
                negative_fix="VALUE_SHIFT",
                #negative_fix="ABS_REV_ACTV",
                debug=False,
                ignore_unknown_class=True
            )
            print(openmax_prob_dict[key].shape)
        openmax_models_predictions[model_idx] = openmax_prob_dict

def cluster_model():

    BATCH_SIZE = 32

    logger.info(f"Using Cluster model with {BATCH_SIZE}")

    cluster_training_dataset= EMNIST(
            dataset_root="./downloads/",
            which_set="train",
            include_unknown=False,
            has_garbage_class=False,
            use_clusters=True,
            num_clusters_per_class=5
        )

    cluster_train_data_loader = torch.utils.data.DataLoader(
        cluster_training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True
    )

    cluster_model = LeNet(
        use_classification_layer=True, use_BG=False, num_classes=50, final_layer_bias=True
    )

    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cluster_model.parameters(), lr=learning_rate, momentum=0.9)


    path_model = "openmax_cnn_eminst_cluster20.pth"

    training_features_dict = train(cluster_model, cluster_training_dataset, cluster_train_data_loader, optimizer, loss_fn, 50, path_model )

if __name__ == "__main__":
    cluster_model()


