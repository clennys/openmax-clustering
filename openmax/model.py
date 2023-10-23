import torch.nn as nn
from loguru import logger
import torch

def cluster_softmax(logits, num_clusters_per_class, labels):
    maxes = torch.max(logits, 1, keepdim=True)[0]
    logits_exp = torch.exp(logits - maxes)
    
    labels //= num_clusters_per_class
    mask = torch.ones_like(logits_exp)
    
    mask_exclude = torch.stack([torch.arange(start * num_clusters_per_class, start * num_clusters_per_class + num_clusters_per_class) for start in labels])
    mask[torch.arange(logits.shape[0]), mask_exclude.T] = 0
    
    x_exp_sum = torch.sum(logits_exp * mask, 1, keepdim=True)
    
    probs = logits_exp / x_exp_sum
    return probs
    

class LeNet(nn.Module):
    def __init__(
        self,
        use_classification_layer=True,
        use_BG=False,
        num_classes=10,
        final_layer_bias=True,
        use_cluster = False,
        num_clusters = 1
    ):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1, padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=50,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.fc1 = nn.Linear(
            in_features=self.conv2.out_channels * 7 * 7, out_features=500, bias=True
        )
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=500, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(
                    in_features=500, out_features=num_classes, bias=final_layer_bias
                )
        self.relu_act = nn.ReLU()
        self.use_classification_layer = use_classification_layer
        logger.info(
            f"{' Model Architecture '.center(90, '#')}\n{self}\n{' Model Architecture End '.center(90, '#')}"
        )

        self.use_cluster = use_cluster
        self.num_clusters = num_clusters

    def forward(self, x, y):
        act_conv1 = self.pool(self.relu_act(self.conv1(x)))
        act_conv2 = self.pool(self.relu_act(self.conv2(act_conv1)))
        act_conv2_reshaped = act_conv2.view(
            -1, self.conv2.out_channels * 7 * 7
        )  #  view method is used to reshape a tensor without changing its data.
        features = self.fc1(act_conv2_reshaped)
        if self.use_classification_layer:
            logits = self.fc2(features)
            if self.use_cluster:
                predictions = cluster_softmax(logits, self.num_clusters, y)
            else:
                predictions = nn.Softmax(dim=1)(logits)
            return predictions, logits, features
        return features
