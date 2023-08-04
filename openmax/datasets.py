from numpy._typing import NDArray
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import Bunch


# Docs: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
class HandwrittenDigits:
    def __init__(self) -> None:
        self.__data: Bunch = load_digits()

    def get_features(self) -> NDArray:
        return self.__data.data

    def get_label(self) -> NDArray:
        return self.__data.target

    def get_feature_names(self) -> list:
        return self.__data.feature_names

    def get_label_names(self) -> list:
        return self.__data.target_names

    def get_raw_image(self, index: int) -> NDArray:
        return self.__data.images[index]

    def plot_image_pdf(self, index: int) -> None:
        image: NDArray = self.get_raw_image(index)
        label: int = self.get_label()[index]
        plt.gray()
        plt.matshow(image)
        # digits_{label}_{index}.pdf
        plt.savefig("digits_{gt}_{idx}.pdf".format(gt=label, idx=index))
