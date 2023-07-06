# Duality in MV-RKM
Primal/Dual Multi-view Restricted Kernel Machine


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

