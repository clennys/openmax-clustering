from openmax_clustering.util.util import *
from openmax_clustering.util.Hyperparameters import *
from openmax_clustering.models.base_model.main import baseline_model
from openmax_clustering.models.cluster_model.main import cluster_model
from loguru import logger
import sys


def logger_setup(debug_output: bool, model_name, path_dir):
    logger.remove()
    if debug_output:
        logger.add(
            path_dir + model_name + f"_debug_log_{get_current_time_str()}.log",
            level="DEBUG",
        )
    logger.add(sys.stderr, level="INFO")


def main():
    args = args_setup().parse_args()
    params = Hyperparameters(args.filename)
    logger_setup(params.logger_output, params.type, params.log_dir)
    logger.info(params.summary())
    torch.set_default_dtype(torch.float32)
    if params.type == "base":
        baseline_model(params, args.gpu)
    else:
        cluster_model(params, args.gpu)
