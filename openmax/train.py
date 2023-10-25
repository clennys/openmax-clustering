import torch
import os
from tqdm import tqdm
from loguru import logger


def train(
    model,
    training_data,
    train_data_loader,
    optimizer,
    loss_fn,
    num_epochs,
    path_model="",
    cluster_per_class=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
        logger.info(f"Loaded: {path_model}")


    model = model.to(device)  # Move model to GPU

    features_dict = {}

    model.train()
    for epoch in range(num_epochs):
        with tqdm(train_data_loader, unit="batch") as tepoch:
            correct_predictions = 0
            correct_predictions_cluster = 0
            total_loss = 0.0

            for batch_inputs, batch_labels in tepoch:
                tepoch.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(
                    device
                )

                optimizer.zero_grad()
                training_predictions, _, training_features = model(
                    batch_inputs
                )

                if epoch == num_epochs - 1:
                    _, training_pred = torch.max(training_predictions, 1)
                    for pred, label, features in zip(
                        training_pred, batch_labels, training_features
                    ):
                        if pred == label:
                            if label.item() in features_dict:
                                features_dict[label.item()] = torch.cat(
                                    (features_dict[label.item()], features[None, :]), 0
                                )
                            else:
                                features_dict[label.item()] = features[None, :]

                loss = loss_fn(training_predictions, batch_labels)

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(training_predictions, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                non_cluster_predicted = torch.div(predicted, 3, rounding_mode='floor').int()
                non_cluster_batch_labels = torch.div(batch_labels, 3, rounding_mode='floor').int()
                correct_predictions_cluster += (
                    (
                        non_cluster_predicted
                        == non_cluster_batch_labels
                    )
                    .sum()
                    .item()
                )
                logger.debug(f"TRAIN PRE: Pred: {predicted}, Batch: {batch_labels}")
                logger.debug(f"TRAIN POST: Pred: {non_cluster_predicted}, Batch: {non_cluster_batch_labels}")

                curr_acc = correct_predictions / len(training_data)
                tepoch.set_postfix(loss=loss.item(), acc=curr_acc)

            accuracy = correct_predictions / len(training_data)
            accuracy_cluster = correct_predictions_cluster / len(training_data)
            avg_loss = total_loss / len(train_data_loader)
            logger.info(
                f"Average loss: {avg_loss:.3f} - Accuracy w/ cluster: {accuracy:.3f} - Accuray w\ cluster: {accuracy_cluster:.3f}"
            )

    # Save the trained model
    torch.save(model.state_dict(), path_model)
    logger.info(f"Saved model: {path_model}")
    return features_dict
