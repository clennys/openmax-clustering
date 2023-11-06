from loguru import logger
from util.util import *
from util.Hyperparameters import *
from models.base_model.main import baseline_model
from models.cluster_model.main import cluster_model




if __name__ == "__main__":
    args = args_setup().parse_args()
    params = Hyperparameters(args.filename)
    logger_setup(params.logger_output, params.log_dir)
    if params.type == "base":
        baseline_model(params, args.gpu)
    elif params.type == "input-cluster":
        cluster_model(params, args.gpu, True, False)
    elif params.type == "feature-cluster":
        cluster_model(params, args.gpu, False, True)
    elif params.type == "input-feature-cluster":
        cluster_model(params, args.gpu, True, True)
    else:
        raise Exception(f"Model Type {params.type} not found, select from 'base', 'input-cluster' or 'feature-cluster'.")
