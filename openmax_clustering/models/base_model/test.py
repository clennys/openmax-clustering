import torch
import os
from tqdm import tqdm
from loguru import logger


def test(model, test_data_loader, test_data, path_model="", device=None):
    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
        logger.info(f"Loaded: {path_model}")

    model = model.to(device) 

    val_logits_dict = {}
    val_features_dict = {}

    with torch.no_grad():
        model.eval()
        with tqdm(test_data_loader, unit="batch") as tepoch:
            correct_predictions = 0

            for batch_inputs, batch_labels in tepoch:
                tepoch.set_description(f"Testing")

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(
                    device
                )

                val_predictions, val_logits, val_features = model(batch_inputs)

                _, val_predicted = torch.max(val_predictions, 1)
                for _, label, logits, features in zip(
                    val_predicted, batch_labels, val_logits, val_features
                ):
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

                correct_predictions += (val_predicted == batch_labels).sum().item()

                batch_acc = correct_predictions / len(test_data)
                tepoch.set_postfix(acc=batch_acc)

    return val_features_dict, val_logits_dict
