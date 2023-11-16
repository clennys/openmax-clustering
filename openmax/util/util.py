import argparse


def args_setup():
    parser = argparse.ArgumentParser(
        prog="CNN with MNIST and OpenMax",
        description="TODO: runs models",
        epilog="TODO: May the force be with you.",
    )

    parser.add_argument("--gpu", "-g", type=int, help="GPU index")

    parser.add_argument("filename", help="Path to config yaml file")

    return parser
