import torch
import torch.nn as nn
from datasets.emnsit import EMNIST
from models.base_model.train import *
from models.base_model.test import *
from models.base_model.validation import *
from models.base_model.model import LeNet
from openset.openmax import *
from openset.metrics import *
from loguru import logger


def baseline_model(params, gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = params.batch_size
    EPOCHS = params.epochs

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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=params.momentum
    )

    path_model = params.saved_models_dir + "openmax_cnn_eminst0.pth"

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

    if not params.train_only:
        val_features_dict, val_logits_dict = validation(
            model, val_data_loader, validation_data, loss_fn, path_model, device=device
        )

        test_features_dict, test_logits_dict = testing(
            model, test_data_loader, test_data, loss_fn, path_model, device=device
        )

        tail_sizes = params.tail_sizes
        logger.info(f"openmax: tail_sizes {tail_sizes}")

        distance_multpls = params.distance_multipls
        logger.info(f"openmax: distance_multpls {distance_multpls}")

        normalize_factor = params.normalize_factor

        for alpha in params.alphas:
            for negative_fix in params.negative_fix:
                logger.info(f"Negative shift: {negative_fix}")
                (
                    _,
                    _,
                    openmax_predictions_per_model,
                    openmax_scores_per_model,
                ) = openmax_run(
                    tail_sizes,
                    distance_multpls,
                    training_features_dict,
                    test_features_dict,
                    test_logits_dict,
                    alpha,
                    negative_fix,
                    normalize_factor,
                )

                acc_per_model = known_unknown_acc(openmax_predictions_per_model, alpha)

                ccr_fpr_per_model = oscr(openmax_scores_per_model)

                save_oscr_values(
                    params.experiment_data_dir,
                    "base",
                    ccr_fpr_per_model,
                    alpha,
                    negative_fix,
                    acc_per_model,
                    normalize_factor
                )
