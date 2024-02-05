# OpenMax-Clustering
This project was created for my Bachelor thesis, called "OpenMax with Clustering for Open-Set Classification". It explores the integration of clustering techniques and optimization to OpenMax in order to enhance its capability in open-set classification.

## Setup
Use the conda install script:

```bash
conda env create -f environment.yaml
```
Activate conda environment with:
```bash
conda activate openmax-clustering
```
## Usage

### Run Model
Use the following command to run the model, the options are set via the configuration file (see below).
```bash
openmax-clustering config/config.yaml --gpu <gpu-index>
```

### Config file
Below you can find the possible options for this project.
*DISCLAIMER*: Be careful by not adding to many options to the lists, otherwise you could run out of memory depending on your maschine. This project is not memory optimized.

```yaml
type: base # base, input-cluster, validation-features-cluster, training-features-cluster, input-validation-features-cluster, input-training-features-cluster
dataset: EMNIST # EMNIST, CIFAR
learning_rate: 0.01
momentum: 0.9
epochs: 1
batch_size: 32
num_clusters_per_class_input: [2]
num_clusters_per_class_features: [3]
alphas: [3] #  -1 removes alpha
tail_sizes: [10, 100, 250, 500, 750, 1000]
distance_multipls: [1.0, 1.25, 1.5, 1.7, 2.0]
negative_fix: ['VALUE_SHIFT'] # ORIGINAL, VALUE_SHIFT, ADJUSTED_NEGATIVE_VALUE
normalize_factor: NONE # NONE, WEIGHTS, N-CLASSES
logger_output: true
run_model: true # Do training, validation and testing
post_process: true # Apply OpenMax
precomputed_clusters: false # Need seperate script to cluster
log_dir: ./logs/
saved_models_dir: ./saved_models/pytorch_models/
saved_network_output_dir: ./saved_models/network_outputs/
experiment_data_dir: ./experiment_data/
clusters_dir: ./saved_models/clusters/
emnist_dir: ./downloads/
thresholds: [0.6, 0.7, 0.9, 1.0] # Used for sigma score
```

## Credits
- [Various Algorithms & Software Tools (VAST)](https://github.com/Vastlab/vast/tree/main)
- [Openset-imagenet-comparison](https://github.com/AIML-IfI/openset-imagenet-comparison)
