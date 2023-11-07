import torch
import os
from tqdm import tqdm
from loguru import logger


def validation(
    model,
    validation_data,
    val_data_loader,
    num_cluster_per_class_input,
    path_model,
    feature_clustering,
    device,
):
    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
        logger.info(f"Loaded: {path_model}")

    model = model.to(device)  # Move model to GPU

    val_logits_dict = {}
    val_features_dict = {}

    with torch.no_grad():
        model.eval()
        with tqdm(val_data_loader, unit="batch") as tepoch:
            correct_predictions = 0
            total_loss = 0.0

            for batch_inputs, batch_labels in tepoch:
                tepoch.set_description(f"Validation")

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(
                    device
                )

                val_predictions, val_logits, val_features = model(batch_inputs)

                _, val_predicted = torch.max(val_predictions, 1)
                for pred, label, logits, features in zip(
                    val_predicted, batch_labels, val_logits, val_features
                ):
                    if not feature_clustering or pred == label:
                        if label.item() in val_features_dict:
                            val_features_dict[label.item()] = torch.cat(
                                (val_features_dict[label.item()], features[None, :]), 0
                            )
                            val_logits_dict[label.item()] = torch.cat(
                                (val_logits_dict[label.item()], logits[None, :]), 0
                            )
                        else:
                            val_features_dict[label.item()] = features[None, :]
                            val_logits_dict[label.item()] = logits[None, :]

                val_predicted = torch.where(
                    val_predicted != -1,
                    val_predicted.int() // num_cluster_per_class_input,
                    val_predicted,
                )

                correct_predictions += (val_predicted == batch_labels).sum().item()

                # loss = loss_fn(val_predictions, batch_labels)

                # total_loss += loss.item()

                batch_acc = correct_predictions / len(validation_data)
                tepoch.set_postfix(acc=batch_acc)

            accuracy = correct_predictions / len(validation_data)
            avg_loss = total_loss / len(val_data_loader)
            logger.info(f"Average loss: {avg_loss:.3f} - Accuracy: {accuracy:.3f}")
    return val_features_dict, val_logits_dict
