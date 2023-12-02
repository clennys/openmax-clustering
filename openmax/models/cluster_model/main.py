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
from util.util import *
from loguru import logger
from util.util import tensor_dict_to_cpu


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


def get_type(params):
    model_type = params.type
    if model_type == "validation-features-cluster":
        input_clustering, validation_features_cluster, training_features_clustering = (
            False,
            True,
            False,
        )
    elif model_type == "training-features-cluster":
        input_clustering, validation_features_cluster, training_features_clustering = (
            False,
            False,
            True,
        )
    elif model_type == "input-cluster":
        input_clustering, validation_features_cluster, training_features_clustering = (
            True,
            False,
            False,
        )
    elif model_type == "input-validation-features-cluster":
        input_clustering, validation_features_cluster, training_features_clustering = (
            True,
            True,
            False,
        )
    elif model_type == "input-training-features-cluster":
        input_clustering, validation_features_cluster, training_features_clustering = (
            True,
            False,
            True,
        )
    else:
        raise Exception("Undefined Model-Cluster Type")

    return input_clustering, validation_features_cluster, training_features_clustering


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


def cluster_model(params, gpu):
    (
        input_clustering,
        validation_features_clustering,
        training_features_clustering,
    ) = get_type(params)

    for n_clusters_per_class_input in params.num_clusters_per_class_input:
        total_n_clusters = n_clusters_per_class_input * 10 if input_clustering else 10

        if input_clustering:
            model_name = f"openmax_cnn_eminst_cluster-{total_n_clusters}"
        else:
            model_name = "openmax_cnn_eminst0"
        path_model = params.saved_models_dir + model_name + ".pth"

        if not params.eval_only:
            device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

            if input_clustering:
                train_data, validation_data, test_data = init_datasets(
                    params, n_clusters_per_class_input
                )
            else:
                train_data, validation_data, test_data = init_datasets(params, 1)

            (
                train_data_loader,
                validation_data_loader,
                test_data_loader,
            ) = init_dataloader(
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

            training_features_dict = train(
                cluster_model,
                train_data,
                train_data_loader,
                optimizer,
                loss_fn,
                params.epochs,
                path_model,
                n_clusters_per_class_input,
                device,
            )

            if not params.train_only:
                val_features_dict, val_logits_dict = validation(
                    cluster_model,
                    validation_data,
                    validation_data_loader,
                    n_clusters_per_class_input,
                    path_model,
                    validation_features_clustering,
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
                saved_output_dict = (
                    training_features_dict,
                    val_features_dict,
                    val_logits_dict,
                    test_features_dict,
                    test_logits_dict,
                )

                network_output_to_pkl(
                    saved_output_dict, params.saved_network_output_dir, model_name + "SPECIAL"
                )

        else:
            (
                training_features_dict,
                val_features_dict,
                val_logits_dict,
                test_features_dict,
                test_logits_dict,
            ) = load_network_output(params.saved_network_output_dir, model_name)

        tail_sizes = params.tail_sizes
        distance_multpls = params.distance_multipls
        negative_fix = params.negative_fix[0]
        normalize_factor = params.normalize_factor

        openmax_training_data = (
            val_features_dict
            if validation_features_clustering
            else training_features_dict
        )

        for (
            n_clusters_per_class_features
        ) in params.num_clusters_per_class_features:
            for alpha in params.alphas:
                (
                    _,
                    _,
                    openmax_predictions_per_model,
                    openmax_scores_per_model,
                ) = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    tensor_dict_to_cpu(openmax_training_data),
                    tensor_dict_to_cpu(test_features_dict),
                    tensor_dict_to_cpu(test_logits_dict),
                    alpha,
                    negative_fix,
                    normalize_factor,
                    n_clusters_per_class_input,
                    n_clusters_per_class_features,
                    training_features_clustering
                )

                acc_per_model = known_unknown_acc(
                    openmax_predictions_per_model, alpha
                )

                preprocess_ccr_fpr = wrapper_preprocess_oscr(
                    openmax_scores_per_model
                )

                ccr_fpr_per_model = oscr(preprocess_ccr_fpr)

                gamma_score = oscr_confidence(preprocess_ccr_fpr)

                epsilon_score = oscr_epsilon_metric(
                    ccr_fpr_per_model, params.thresholds
                )

                results_dict = {
                    "ACC": acc_per_model,
                    "CCR-FPR": ccr_fpr_per_model,
                    "GAMMA": gamma_score,
                    "EPSILON": epsilon_score,
                    "ALPHA": alpha,
                    "N-FIX": negative_fix,
                    "MODEL-TYPE": params.type,
                    "NORM-FACTOR": normalize_factor,
                    "INPUT-CLUSTER": n_clusters_per_class_input,
                    "FEATURES-CLUSTER": n_clusters_per_class_features,
                    "TAILSIZES": tail_sizes,
                    "DIST-MULT": distance_multpls,
                }

                save_oscr_values(params.experiment_data_dir, results_dict)
