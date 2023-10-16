import torch
import os
from tqdm import tqdm

def validation(model, val_data_loader, validation_data, loss_fn, path_model=""):
    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model))
        print(f"Loaded: {path_model}")

    val_logits_dict = {}
    val_features_dict = {}

    with torch.no_grad():
        model.eval()
        with tqdm(val_data_loader, unit="batch") as tepoch:
            correct_predictions = 0
            total_loss = 0.0

            for batch_inputs, batch_labels in tepoch:
                tepoch.set_description(f"Validation")

                val_predictions, val_logits, val_features = model(batch_inputs)

                _, val_predicted = torch.max(val_predictions, 1)
                for pred, label, logits, features in zip(
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

                # loss = loss_fn(val_predictions, batch_labels)

                # total_loss += loss.item()

                batch_acc = correct_predictions / len(validation_data)
                tepoch.set_postfix(acc=batch_acc)

            accuracy = correct_predictions / len(validation_data)
            avg_loss = total_loss / len(val_data_loader)
            print(f"Average loss: {avg_loss:.3f} - Accuracy: {accuracy:.3f}")
    return val_features_dict, val_logits_dict

