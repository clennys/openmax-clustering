import torch.nn as nn
from loguru import logger
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
        self,
        use_classification_layer=True,
        use_BG=False,
        num_classes=10,
        final_layer_bias=True,
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

    def forward(self, x):
        act_conv1 = self.pool(self.relu_act(self.conv1(x)))
        act_conv2 = self.pool(self.relu_act(self.conv2(act_conv1)))
        act_conv2_reshaped = act_conv2.view(
            -1, self.conv2.out_channels * 7 * 7
        )  #  view method is used to reshape a tensor without changing its data.
        features = self.fc1(act_conv2_reshaped)
        if self.use_classification_layer:
            logits = self.fc2(features)
            predictions = nn.Softmax(dim=1)(logits)
            return predictions, logits, features
        return features

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        features = out
        logits = self.linear(features)
        predictions = nn.Softmax(dim=1)(logits)
        return predictions, logits, features


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes) 