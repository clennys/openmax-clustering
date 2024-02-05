import argparse
import torch
from datetime import datetime
from loguru import logger
import pickle
import io


def args_setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", "-g", type=int, help="GPU index")

    parser.add_argument("filename", help="Path to config yaml file")

    return parser


def tensor_dict_to_cpu(tensors_dict):
    if torch.cuda.is_available():
        for key in tensors_dict:
            tensors_dict[key] = tensors_dict[key].cpu()
    return tensors_dict


def get_current_time_str():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    return dt_string


def network_output_to_pkl(data, path, model_name, special=False):
    file_ = path + "dnn_output_" + f"{model_name}" + ".pkl"

    with open(file_, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Network output saved as {file_}.")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_network_output(path, model_name, special=False):
    file_ = path + "dnn_output_" + f"{model_name}" + ".pkl"

    with open(file_, "rb") as f:
        loaded_file = CPU_Unpickler(f).load()
        return loaded_file


def cluster_output_to_pkl(data, path, model_name, n_clusters, training_clustering):
    type_c = "training" if training_clustering else "validation"
    file_ = path + "dnn_cluster_" + f"{model_name}_{type_c}_{n_clusters}" + ".pkl"
    with open(file_, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Saved:{file_}")


def load_cluster_output(path, model_name, n_clusters, training_clustering):
    type_c = "training" if training_clustering else "validation"
    file_ = path + "dnn_cluster_" + f"{model_name}_{type_c}_{n_clusters}" + ".pkl"
    with open(file_, "rb") as f:
        loaded_file = CPU_Unpickler(f).load()
        logger.info(f"Loaded: {file_}")
        return loaded_file
