import yaml
import loguru as logger

class Params:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.__dict__.update(params)

def logger_setup(debug_output: bool, path_dir):
    logger.remove()
    if debug_output:
        logger.add(path_dir + "debug_log_{time}.log", level="DEBUG")
    logger.add(sys.stderr, level="INFO")


def args_setup():
    parser = argparse.ArgumentParser(
        prog="CNN with MNIST and OpenMax",
        description="TODO: runs models",
        epilog="TODO: May the force be with you.",
    )

    parser.add_argument(
        "--gpu", "-g", type=int, help="GPU index"
    )

    parser.add_argument('filename', help="Path to config yaml file")

    return parser