import argparse
import torch


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
