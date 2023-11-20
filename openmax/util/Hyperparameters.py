import yaml
import loguru as logger


class Hyperparameters:
    def __init__(self, file_path):
        with open(file_path, "r") as file:
            params = yaml.safe_load(file)
        self.__dict__.update(params)

    def summary(self):
        summary_str = ""
        for key in self.__dict__.keys():
            summary_str += f"{key}:\t{self.__dict__[key]}\n"

        return f"{' Selected Parameters '.center(90, '#')}\n{summary_str}\n{' Selected Parameters End '.center(90, '#')}"

