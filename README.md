This repository contains the source code used in [*Roto-Translation Equivariant Super-Resolution of Two-Dimensional Flows Using Convolutional Neural Networks*](https://arxiv.org/abs/2202.11099).

- [Setup](#setup)
- [Docker containers](#docker-containers)
  - [Fortran](#fortran)
  - [PyTorch](#pytorch)
- [Singularity containers](#singularity-containers)

# Setup

1. Install [docker](https://www.docker.com/) and confirm that `docker-compose` works.
2. Build the containers: `$ docker-compose build`
3. Start the containers: `$ docker-compose up -d`

# Docker containers

## Fortran

- The Fortran container is used to perform the fluid simulations
- The simulations can be conducted as follows
```
$ ./script/conduct_fortran_experiments.sh
```

## PyTorch

- The PyTorch container is used to train the models and to analyze data.
  - The JupyterLab can be utilized: [localhost:8888](http://localhost:8888/)

# Singularity containers

- Singularity containers are used to train models on [TSUBAME3](https://www.gsic.titech.ac.jp/en)
- The container is built as follows
```
$ singularity build -f pytorch.sif ./singularity/pytorch_tsubame/pytorch.def
```