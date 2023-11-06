import yaml
import loguru as logger

class Hyperparameters:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.__dict__.update(params)

