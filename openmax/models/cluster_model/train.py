import torch
import os
from tqdm import tqdm
from loguru import logger


def train(
    model,
    sampler,
    train_data_loader,
    optimizer,
    loss_fn,
    num_epochs,
    path_model,
    input_clustering_num,
    device,
    input_training_cluster=False,
):
    if os.path.isfile(path_model):
        model.load_state_dict(torch.load(path_model, map_location=device))
        logger.info(f"Loaded: {path_model}")

    model = model.to(device)

    features_dict = {}

    input_clustering = input_clustering_num > 1

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
                training_predictions, _, training_features = model(batch_inputs)

                if input_clustering and epoch == num_epochs - 1:
                    _, training_pred = torch.max(training_predictions, 1)
                    for pred, label, features in zip(
                        training_pred, batch_labels, training_features
                    ):
                        if input_training_cluster and False:
                            pred = pred // input_clustering_num
                            label = label // input_clustering_num
                        if pred == label:
                            if label.item() in features_dict:
                                features_dict[label.item()] = torch.cat(
                                    (features_dict[label.item()], features[None, :]), 0
                                )
                            else:
                                features_dict[label.item()] = features[None, :]

                loss = loss_fn(training_predictions, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(training_predictions, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                if input_clustering:
                    non_cluster_predicted = torch.div(
                        predicted, input_clustering_num, rounding_mode="floor"
                    ).int()
                    non_cluster_batch_labels = torch.div(
                        batch_labels, input_clustering_num, rounding_mode="floor"
                    ).int()
                    correct_predictions_cluster += (
                        (non_cluster_predicted == non_cluster_batch_labels).sum().item()
                    )

                curr_acc = correct_predictions / len(sampler)
                tepoch.set_postfix(loss=loss.item(), acc=curr_acc)

            accuracy = correct_predictions / len(sampler)
            accuracy_cluster = correct_predictions_cluster / len(sampler)
            avg_loss = total_loss / len(train_data_loader)

            if input_clustering:
                logger.info(
                    f"Average loss: {avg_loss:.3f} - Accuracy w/ cluster: {accuracy:.3f} - Accuray w\ cluster: {accuracy_cluster:.3f}"
                )
            else:
                logger.info(f"Average loss: {avg_loss:.3f} - Accuracy {accuracy:.3f}")

    # Save the trained model
    torch.save(model.state_dict(), path_model)
    logger.info(f"Saved model: {path_model}")
    return features_dict
