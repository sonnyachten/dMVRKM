# Duality in Multi-View Restricted Kernel Machines
## Useful links
[Project page](www.sonnyachten.com/dMVRKM)

[Paper](www.google.comhttps://arxiv.org/abs/2305.17251)

## Abstract
We propose a unifying setting that combines existing restricted kernel machine methods into a single primal-dual multi-view framework for kernel principal component analysis in both supervised and unsupervised settings. We derive the primal and dual representations of the framework and relate different training and inference algorithms from a theoretical perspective. We show how to achieve full equivalence in primal and dual formulations by rescaling primal variables. Finally, we experimentally validate the equivalence and provide insight into the relationships between different methods on a number of time series data sets by recursively forecasting unseen test data and visualizing the learned features.

## Usage

Install miniconda: ` https://docs.conda.io/en/latest/miniconda.html `

### Install python packages in conda environment

```
conda env create -f environment.yml
```

### Train
Activate the conda environment `conda activate rkm_env` and run one of the following commands, for example:
```
python3 main.py
```
The above runs a general base code for further experimentation. To reproduce the
results from the paper, follow the instructions below.

### Reproduce the experiments

#### Sum of Sinusoidal waves
```
python3 main.py --config-name sine_sum_eig_cfg.yaml
```

```
python3 main.py --config-name sine_sum_stiefel_cfg.yaml hyperparameters.mode=primal
```

```
python3 main.py --config-name sine_sum_stiefel_cfg.yaml hyperparameters.mode=mode
```

#### SantaFe
```
python3 main.py --config-name santafe_eig_cfg.yaml
```

```
python3 main.py --config-name santafe_stiefel_cfg.yaml
```

To quickly test a pre-trained model:
```
python3 main.py --config-name santafe_stiefel_cfg.yaml +hyperparameters.pre_trained_model_path=outputs/10-08-41/model_stiefel.pt;
```

