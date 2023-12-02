import torch
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class CIFAR(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        dataset_root,
        which_set="train",
        include_unknown=True,
    ):
        transformation = transform_train if which_set == "train" else transform_test
        self.CIFAR10 = torchvision.datasets.CIFAR10(root=dataset_root, train=which_set == "train", download=True, transform = transformation)
        self.CIFAR100 = torchvision.datasets.CIFAR100(root=dataset_root, train=which_set == "train", download=True, transform=transformation) if include_unknown else list()

        self.which_set = which_set

    def __getitem__(self, index):
            if index < len(self.CIFAR10):
                return self.CIFAR10[index]
            else:
                return (
                    self.CIFAR100[index - len(self.CIFAR10)][0],  -1,
                )

    def __len__(self):
        return len(self.CIFAR10) + len(self.CIFAR100)




