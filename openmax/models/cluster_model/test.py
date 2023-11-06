import torch
import os
from tqdm import tqdm
from loguru import logger


def testing(
    model,
    test_data,
    test_data_loader,
    loss_fn,
    num_cluster_per_class_input,
    path_model,
    device,
):

    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
        logger.info(f"Loaded: {path_model}")

    model = model.to(device)

    test_logits_dict = {}
    test_features_dict = {}

    with torch.no_grad():
        model.eval()
        with tqdm(test_data_loader, unit="batch") as tepoch:
            correct_predictions = 0
            total_loss = 0.0

            for batch_inputs, batch_labels in tepoch:
                tepoch.set_description(f"Testing")

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(
                    device
                )

                test_predictions, test_logits, test_features = model(batch_inputs)

                _, test_predicted = torch.max(test_predictions, 1)

                for _, label, logits, features in zip(
                    test_predicted, batch_labels, test_logits, test_features
                ):
                    if label.item() in test_features_dict:
                        test_features_dict[label.item()] = torch.cat(
                            (test_features_dict[label.item()], features[None, :]), 0
                        )
                        test_logits_dict[label.item()] = torch.cat(
                            (test_logits_dict[label.item()], logits[None, :]), 0
                        )
                    else:
                        test_features_dict[label.item()] = features[None, :]
                        test_logits_dict[label.item()] = logits[None, :]

                test_predicted = torch.where(
                    test_predicted != -1,
                    test_predicted.int() // num_cluster_per_class_input,
                    test_predicted,
                )

                correct_predictions += (test_predicted == batch_labels).sum().item()

                # loss = loss_fn(val_predictions, batch_labels)

                # total_loss += loss.item()

                batch_acc = correct_predictions / len(test_data)
                tepoch.set_postfix(acc=batch_acc)

            accuracy = correct_predictions / len(test_data)
            avg_loss = total_loss / len(test_data_loader)
            logger.info(f"Average loss: {avg_loss:.3f} - Accuracy: {accuracy:.3f}")
    return test_features_dict, test_logits_dict