# Openset Imagenet Flow

## Training

### Train.py

Entry point for network training with openmax.

```python train.py
def main(command_line_options = None):

    args = get_args(command_line_options)
    config = openset_imagenet.util.load_yaml(args.configuration)

    if args.gpu is not None:
        config.gpu = args.gpu
    config.protocol = args.protocol

    if config.algorithm.type == "threshold":
        openset_imagenet.train.worker(config)
    elif config.algorithm.type in ['openmax', 'evm']: # ---> HERE
        openset_imagenet.openmax_evm.worker(config)
    elif config.algorithm.type == "proser":
        openset_imagenet.proser.worker(config, 0)
    else:
        raise ValueError(f"The training configuration type '{config.algorithm.type}' is not known to the system")
```

### Openmax_evm.py

```python openmax_evm.py
def worker(cfg):
    """ Main worker creates all required instances, trains and validates the model.
    Args:
        cfg (NameSpace): Configuration of the experiment
    """

    # OMITTED: SETUP CODE FOR MODEL AND CHECKPOINTS --> Check source file

    if cfg.algorithm.type== 'openmax':
        hyperparams = openmax_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.distance_multiplier, cfg.algorithm.translateAmount, cfg.algorithm.distance_metric, cfg.algorithm.alpha)
    elif cfg.algorithm.type == 'evm':
        hyperparams = evm_hyperparams(cfg.algorithm.tailsize, cfg.algorithm.cover_threshold,cfg.algorithm.distance_multiplier, cfg.algorithm.distance_metric, cfg.algorithm.chunk_size)

    logger.info("Feature extraction on training data:")

    # extracting arrays for training data
    gt, logits, features, scores = get_arrays(
            model=model,
            loader=train_loader,
            garbage=cfg.loss.type=="garbage",
            pretty=not cfg.parallel
    )


    gt, features, logits = torch.Tensor(gt)[:, None], torch.Tensor(features), torch.Tensor(logits)

    targets, features, logits = postprocess_train_data(gt, features, logits)
    pos_classes = collect_pos_classes(targets)

    feat_dict, _ = compose_dicts(targets, features, logits)

    logger.debug('\n')
    logger.info(f'Starting {cfg.algorithm.type} Training Procedure:')

    training_fct = get_training_function(cfg.algorithm.type)

    #performs training on all parameter combinations
    #Training method returns iterator over (hparam_combo, (class, {model}))
    all_hyper_param_models = list(training_fct(
        pos_classes_to_process=pos_classes, features_all_classes=feat_dict, args=hyperparams, gpu=cfg.gpu, models=None))

    save_models(all_hyper_param_models, pos_classes, cfg)
    logger.info(f'{cfg.algorithm.type} Training Finished')
```

1. Get hyperparams
   - Why and what is NameSpace?

```python
def openmax_hyperparams(tailsize, dist_mult, translate_amount, dist_metric, alpha):

    return NameSpace(dict(
        tailsize=tailsize, distance_multiplier=dist_mult, distance_metric=dist_metric, alpha=alpha
    ))
```

2. Exract Arrays for the training data
3. Load onto tensor
4. Postprocess the train data

```python
def postprocess_train_data(targets, features, logits):
    # Note: OpenMax uses only the training samples that get correctly classified by the
          # underlying, extracting DNN to train its model.logger.debug('\n')

    with torch.no_grad():
        # OpenMax only uses KKCs for training
        known_idxs = (targets >= 0).squeeze()

        targets_kkc, features_kkc, logits_kkc = targets[
            known_idxs], features[known_idxs], logits[known_idxs]

        class_predicted = torch.max(logits_kkc, axis=1).indices
        correct_idxs = targets_kkc.squeeze() == class_predicted

        logger.info(
            f'Correct classifications: {torch.sum(correct_idxs).item()}')
        logger.info(
            f'Incorrect classifications: {torch.sum(~correct_idxs).item()}')
        logger.info(
            f'Number of samples after post-processing: {targets_kkc[correct_idxs].shape[0]}')
        logger.info(
            f'Number of unique classes after post-processing: {len(collect_pos_classes(targets_kkc[correct_idxs]))}')

        return targets_kkc[correct_idxs], features_kkc[correct_idxs], logits_kkc[correct_idxs]
```

5. Collect the position of classes

```python
def collect_pos_classes(targets):
    targets_unique = torch.unique(targets, sorted=True)
    pos_classes = targets_unique[targets_unique >= 0].numpy().astype(np.int32).tolist()
    return pos_classes
```

6. Get Training function
   EVT Meta-Recognition Calibration for Open Set Deep Networks, with per class Weibull fit to $\\nrleg$ largest distance to mean activation vector.
   Returns libMR models $ρ_j$ which includes parameters $\\tau_i$ for shifting the data as well as the Weibull shape and scale parameters: $\\kappa_i$, $\\lambda_i$.
   
![Algo1](https://github.com/devnnys/openmax/blob/main/notes/images/algo1_calib.png?raw=true "Openmax calibration algorithm")

```python
def OpenMax_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the EVM_Params function.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    """
    if "translateAmount" not in args.__dict__:
        args.translateAmount = 1
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name].clone().to(device)
        MAV = torch.mean(features, dim=0).to(device)
        distances = pairwisedistances.__dict__[args.distance_metric](
            features, MAV[None, :]
        )
        for tailsize, distance_multiplier in itertools.product(
            args.tailsize, args.distance_multiplier
        ):
            weibull_model = fit_high(
                distances.T, distance_multiplier, tailsize, args.translateAmount
            )
            yield (
                f"TS_{tailsize}_DM_{distance_multiplier:.2f}",
                (pos_cls_name, dict(MAV=MAV.cpu()[None, :], weibulls=weibull_model)),
            )
```

7. Perform training on all data and combinations
8. Save Model (.pth)
