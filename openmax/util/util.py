import argparse
import torch
from datetime import datetime
from loguru import logger
import pickle


def args_setup():
    parser = argparse.ArgumentParser(
        prog="CNN with MNIST and OpenMax",
        description="TODO: runs models",
        epilog="TODO: May the force be with you.",
    )

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
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    return dt_string


def network_output_to_pkl(data, path, model_name):
    file_ = path + "dnn_output_" + f"{model_name}" + ".pkl"
    with open(file_, "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Network output saved as {file_}.")


def load_network_output(path, model_name):
    file_ = path + "dnn_output_" + f"{model_name}" + ".pkl"
    with open(file_, "rb") as f:
        loaded_file = pickle.load(f)
        return loaded_file
