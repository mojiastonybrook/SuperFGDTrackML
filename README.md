# Notes on SuperFGD Particle Trajectory Fitting with Machine-learning-based Models

This repository is mostly hacked from the one that contains the code implementing the methods described in the paper "*Artificial intelligence for improved fitting of trajectories of elementary particles in inhomogeneous dense materials immersed in a magnetic field*": [[Commun. Phys.]](https://doi.org/10.1038/s42005-023-01239-4) [[arXiv]](https://arxiv.org/abs/2211.04890v1).
It is modified to accomandate the computational environment of the nnhome clusters.

## Contents

    .
    ├── README.md
    ├── jobs
        |── run_train_xx.sh
    ├── trajectory_fitting-main
        ├── fitting_algorithms
        │   ├── __init__.py
        │   ├── rnn.py
        │   ├── sir_pf.py
        │   └── transformer.py
        ├── modules
        │   ├── __init__.py
        │   ├── constants.py
        │   └── dataset.py
        ├── nn_training
        │   ├── __init__.py
        │   ├── train_rnn.py
        │   └── train_transformer.py
        └── models
            └──
 
- README.md: this file.
- jobs: folder containing the shell scripts to submit remote training jobs on a nnhome cluster.
- trajectory_fitting-main: the main package for the ML-based methods; it includes
    - fitting_algorithms: folder containing the implementation of the methods RNN, Transformer, and SIR-PF.
    - modules: folder containing some utility functions used by the fitting algorithms.
    - nn_training: folder containing scripts for training the RNN and Transformer models.
    - models: folder where the trained models are saved; create one if not existing 

## How to install the trajectory fitter on a nnhome cluster

The code was developed using Python 3.10.4 and PyTorch 1.11.0. Set up the python environment by conda.

To install conda, please check for more information.

With conda, install Python 3.10.4 by:
- `conda create --name py3104_env -c anaconda python=3.10.4`

Activate this envrionment by:
-`conda activate py3104_env`

Install PyTorch 1.11.0 with pip:
- `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

Install numpy with pip:
- `pip install numpy==1.26.4`

Inside the `trajectory_fitting-main` directory, create a folder to hold the training outputs:
- `mkdir models`


The datasets can be found on nnhome space here: [Zenodo](https://doi.org/10.5281/zenodo.10998285).

## How to run the code

To run the code, you need to have Python 3.10.4 and PyTorch 1.11.0 installed. The datasets and checkpoints can be downloaded from https://doi.org/10.5281/zenodo.10998285.

The scripts that should be run are:

- `python -m nn_training.train_rnn` to train the RNN model.
- `python -m nn_training.train_transformer` to train the Transformer model.

If you try to run the scripts directly from their folders (e.g., python nn_training/train_rnn.py), you may get an error about an attempted relative import with no known parent package.


