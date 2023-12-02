import torch
import torch.nn as nn
from datasets.emnsit import EMNIST
from datasets.CIFAR import CIFAR
from models.base_model.train import *
from models.base_model.test import *
from models.base_model.validation import *
from models.base_model.model import LeNet, ResNet18
from torch.utils.data.sampler import SubsetRandomSampler
from openset.openmax import *
from openset.metrics import *
from loguru import logger
from util.util import *


def train_val_balanced_samplers(val_ratio, train_dataset):
    n_samples_class = int(np.floor(len(train_dataset) * val_ratio / 10))

    # Get all the targets from the dataset
    targets = np.array(train_dataset.targets)

    # Initialize lists to store the train and validation indices
    train_indices = []
    valid_indices = []

    # For each class
    for i in range(10):  # Assuming there are 10 classes in CIFAR10
        # Get the indices for this class
        class_indices = np.where(targets == i)[0]

        # Randomly permute the indices
        np.random.seed(0)
        np.random.shuffle(class_indices)

        # Split the indices into train and validation indices
        train_indices.extend(class_indices[n_samples_class:])
        valid_indices.extend(class_indices[:n_samples_class])

    # Define samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    return train_sampler, valid_sampler


def init_datasets(params, cluster_per_class=1):
    if params.dataset == "EMNIST":
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
    elif params.dataset == "CIFAR":
        train_data = CIFAR(
            dataset_root=params.emnist_dir,
            which_set="train",
            include_unknown=False,
        )

        validation_data = CIFAR(
            dataset_root=params.emnist_dir,
            which_set="val",
            include_unknown=False,
        )

        test_data = CIFAR(
            dataset_root=params.emnist_dir,
            which_set="test",
            include_unknown=True,
        )
    else:
        raise Exception("Unable to find the dataset.")

    return train_data, validation_data, test_data


def baseline_model(params, gpu):
    if params.dataset == "EMNIST":
        model_name = "openmax_cnn_eminst0"
    else:
        model_name = "openmax_cnn_cifar0"

    if params.run_model:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        training_data, validation_data, test_data = init_datasets(params, 1)

        BATCH_SIZE = params.batch_size
        EPOCHS = params.epochs

        known_train_dataset = training_data.mnist if params.dataset == "EMNIST" else training_data.CIFAR10

        train_sampler, val_sampler = train_val_balanced_samplers(0.2, known_train_dataset)

        train_data_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=BATCH_SIZE,
            # shuffle=True,
            num_workers=5,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_data_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=BATCH_SIZE, pin_memory=True, sampler=val_sampler
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_data, batch_size=BATCH_SIZE, pin_memory=True
        )

        if params.dataset == "EMNIST":
            model = LeNet(
                use_classification_layer=True,
                use_BG=False,
                num_classes=10,
                final_layer_bias=True,
            )
        else:
            model = ResNet18(
                num_classes=10,
            )

        learning_rate = params.learning_rate
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=params.momentum
        )
        path_model = params.saved_models_dir + model_name + ".pth"

        training_features_dict = train(
            model,
            training_data,
            train_data_loader,
            optimizer,
            loss_fn,
            EPOCHS,
            path_model,
            device=device,
        )

        val_features_dict, val_logits_dict = validation(
            model,
            val_data_loader,
            validation_data,
            loss_fn,
            path_model,
            device=device,
        )

        test_features_dict, test_logits_dict = testing(
            model, test_data_loader, test_data, loss_fn, path_model, device=device
        )

        saved_output_dict = (
            training_features_dict,
            val_features_dict,
            val_logits_dict,
            test_features_dict,
            test_logits_dict,
        )

        network_output_to_pkl(
            saved_output_dict, params.saved_network_output_dir, model_name
        )

    else:
        (
            training_features_dict,
            val_features_dict,
            val_logits_dict,
            test_features_dict,
            test_logits_dict,
        ) = load_network_output(params.saved_network_output_dir, model_name)

    if params.post_process:
        tail_sizes = params.tail_sizes
        distance_multpls = params.distance_multipls
        normalize_factor = params.normalize_factor

        for alpha in params.alphas:
            for negative_fix in params.negative_fix:
                (
                    _,
                    _,
                    openmax_predictions_per_model,
                    openmax_scores_per_model,
                ) = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    tensor_dict_to_cpu(training_features_dict),
                    tensor_dict_to_cpu(test_features_dict),
                    tensor_dict_to_cpu(test_logits_dict),
                    alpha,
                    negative_fix,
                    normalize_factor,
                )

                acc_per_model = known_unknown_acc(openmax_predictions_per_model, alpha)

                preprocess_ccr_fpr = wrapper_preprocess_oscr(openmax_scores_per_model)

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
                    "INPUT-CLUSTER": 1,
                    "FEATURES-CLUSTER": 1,
                    "TAILSIZES": tail_sizes,
                    "DIST-MULT": distance_multpls,
                }

                save_oscr_values(params.experiment_data_dir, results_dict)
