This repository contains the source code used in [*Rotationally equivariant super-resolution of velocity fields in two-dimensional flows using convolutional neural networks*](https://doi.org/10.1063/5.0132326).

- [Setup](#setup)
- [Docker containers](#docker-containers)
  - [Fortran](#fortran)
  - [PyTorch](#pytorch)
- [Singularity containers](#singularity-containers)
- [Experiments](#experiments)
  - [How to make train and test data](#how-to-make-train-and-test-data)
  - [How to train models](#how-to-train-models)
- [Cite](#cite)

# Setup

1. Install [docker](https://www.docker.com/) and confirm that `docker-compose` works.
2. Build the containers: `$ docker-compose build`
3. Start the containers: `$ docker-compose up -d`

# Docker containers

## Fortran

- The Fortran container is used to perform the fluid simulations.
  - **The total data size will be about 220 GB.**

## PyTorch

- The PyTorch container is used to train the models and to analyze data.
  - The JupyterLab can be used: [localhost:8888](http://localhost:8888/)

# Singularity containers

- [Singularity](https://sylabs.io/guides/3.0/user-guide/) containers were used to train models on [TSUBAME3](https://www.gsic.titech.ac.jp/en)
- The container can be built as follows
```
$ singularity build -f pytorch.sif ./singularity/pytorch_tsubame/pytorch.def
```

# Experiments

## How to make train and test data

- **The total data size will be about 220 GB.**

1. Conduct the Fortran numerical experiments: `$ ./script/conduct_fortran_experiments.sh`
2. Make the train and test data using each notebook in `./pytorch/notebook`
   1. Start the pytorch container: `$ docker-compose up -d pytorch`
   2. Connect to JupyterLab: [localhost:8888](http://localhost:8888/)
   3. Run each notebook.

## How to train models

1. Start the pytorch container: `$ docker-compose up -d pytorch`
2. Run the script as follows:
  - `experiment_name`: experiment name. See in `./pytorch/config`
  - `data_method`: method of creating data. See in `./pytorch/config`
  - `config_name`: name of configuration. See in `./pytorch/config`

```
$ docker-compose exec pytorch python /workspace/pytorch/script/train_model.py \
  --data_method ${data_method} --experiment_name ${experiment_name} --config_name ${config_name}
#
# An example
#
$ docker-compose exec pytorch python /workspace/pytorch/script/train_model.py \
  --data_method average --experiment_name decaying_turbulence --config_name dscms_scale09_V
```

# Cite

```
@article{
    author = {Yasuda, Yuki and Onishi, Ryo},
    title = {Rotationally equivariant super-resolution of velocity fields in two-dimensional flows using convolutional neural networks},
    journal = {APL Machine Learning},
    volume = {1},
    number = {2},
    year = {2023},
    month = {04},
    issn = {2770-9019},
    doi = {10.1063/5.0132326},
    url = {https://doi.org/10.1063/5.0132326},
    note = {026107},
    eprint = {https://pubs.aip.org/aip/aml/article-pdf/doi/10.1063/5.0132326/16820201/026107\_1\_5.0132326.pdf},
}
```