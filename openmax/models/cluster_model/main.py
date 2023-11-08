import torch
import torch.nn as nn
from datasets.emnsit import EMNIST
from models.cluster_model.train import *
from models.cluster_model.validation import *
from models.cluster_model.test import *
from models.base_model.model import LeNet
from openset.openmax import *
from openset.metrics import *
from util.Hyperparameters import *
from loguru import logger

def init_datasets(params, cluster_per_class=1):
    train_data = EMNIST(
        dataset_root=params.emnist_dir,
        which_set="train",
        include_unknown=False,
        has_garbage_class=False,
        use_clusters=True if cluster_per_class > 1 else False,
        num_clusters_per_class=cluster_per_class,
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

    return train_data, validation_data, test_data


def init_dataloader(train_data, validation_data, test_data, batch_size):
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True,
    )

    val_data_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=batch_size, pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, pin_memory=True
    )

    return train_data_loader, val_data_loader, test_data_loader


def cluster_model(params, gpu, input_clustering, feature_clustering):
    if not input_clustering and not feature_clustering:
        raise Exception(
            "Need to specify at least one clusering option, otherwise select the base model."
        )

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    for n_clusters_per_class_input in params.num_clusters_per_class_input:
        total_n_clusters = n_clusters_per_class_input * 10 if input_clustering else 10

        logger.info(
            f"Using Cluster Model with Input Clustering={input_clustering} and Feature Clustering={feature_clustering}"
        )

        if input_clustering:
            train_data, validation_data, test_data = init_datasets(
                params, n_clusters_per_class_input
            )
        else:
            train_data, validation_data, test_data = init_datasets(params, 1)

        train_data_loader, validation_data_loader, test_data_loader = init_dataloader(
            train_data, validation_data, test_data, params.batch_size
        )

        cluster_model = LeNet(
            use_classification_layer=True,
            use_BG=False,
            num_classes=total_n_clusters,
            final_layer_bias=True,
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            cluster_model.parameters(),
            lr=params.learning_rate,
            momentum=params.momentum,
        )

        if input_clustering:
            path_model = (
                params.saved_models_dir
                + f"openmax_cnn_eminst_cluster-{total_n_clusters}.pth"
            )
        else:
            path_model = params.saved_models_dir + "openmax_cnn_eminst0.pth"

        training_features_dict = train(
            cluster_model,
            train_data,
            train_data_loader,
            optimizer,
            loss_fn,
            params.epochs,
            path_model,
            input_clustering,
            device,
        )

        if not params.train_only:
            val_features_dict, val_logits_dict = validation(
                cluster_model,
                validation_data,
                validation_data_loader,
                n_clusters_per_class_input,
                path_model,
                feature_clustering,
                device,
            )

            test_features_dict, test_logits_dict = testing(
                cluster_model,
                test_data,
                test_data_loader,
                loss_fn,
                n_clusters_per_class_input,
                path_model,
                device,
            )

            tail_sizes = params.tail_sizes
            logger.info(f"openmax: tail_sizes {tail_sizes}")

            distance_multpls = params.distance_multpls
            logger.info(f"openmax: distance_multpls {distance_multpls}")

            negative_fix = params.negative_fix[0]

            openmax_training_data = val_features_dict if feature_clustering else training_features_dict

            for n_clusters_per_class_features in params.num_clusters_per_class_features:
                for alpha in params.alphas:

                    n_cluster_per_class = n_clusters_per_class_features if feature_clustering else n_clusters_per_class_input
                    (
                        _,
                        _,
                        openmax_predictions_per_model,
                        openmax_scores_per_model,
                    ) = openmax_run(
                        tail_sizes,
                        distance_multpls,
                        openmax_training_data,
                        test_features_dict,
                        test_logits_dict,
                        alpha,
                        negative_fix,
                        n_cluster_per_class,
                        feature_clustering
                    )

                    known_unknown_acc(
                        openmax_predictions_per_model, n_clusters_per_class_input
                    )

                    ccr_fpr_per_model = oscr(openmax_scores_per_model)

                    save_oscr_values(
                        params.experiment_data_dir,
                        params.type,
                        n_clusters_per_class_features,
                        ccr_fpr_per_model,
                        alpha,
                        negative_fix,
                    )
