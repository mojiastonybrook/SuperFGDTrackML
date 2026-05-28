# Particle Trajectory Fitting with Artificial Intelligence

This repository contains the code implementing the methods described in the paper "*Artificial intelligence for improved fitting of trajectories of elementary particles in inhomogeneous dense materials immersed in a magnetic field*": [[Commun. Phys.]](https://doi.org/10.1038/s42005-023-01239-4) [[arXiv]](https://arxiv.org/abs/2211.04890v1).

The paper shows how to enhance the resolution of the elementary particle track fitting in inhomogeneous dense detectors, such as plastic scintillators, by using artificial intelligence algorithms. The authors use deep learning to replace more traditional Bayesian filtering methods, drastically improving the reconstruction of the interacting particle kinematics. The specific form of neural network inherited from the field of natural language processing, is very close to the concept of a Bayesian filter that adopts a hyper-informative prior. Such a paradigm change can influence the design of future particle physics experiments and their data exploitation.

The code was developed using Python 3.10.4 and PyTorch 1.11.0. The datasets and checkpoints can be found here: [Zenodo](https://doi.org/10.5281/zenodo.10998285).

## File structure

The file structure of this repository is organized as follows:

    .
    ├── LICENSE
    ├── README.md
    ├── __init__.py
    ├── fitting_algorithms
    │   ├── __init__.py
    │   ├── rnn.py
    │   ├── sir_pf.py
    │   └── transformer.py
    ├── modules
    │   ├── __init__.py
    │   ├── constants.py
    │   └── dataset.py
    └── nn_training
        ├── __init__.py
        ├── train_rnn.py
        └── train_transformer.py


- LICENSE: MIT license file.
- README.md: this file.
- fitting_algorithms: folder containing the implementation of the methods RNN, Transformer, and SIR-PF.
- modules: folder containing some utility functions used by the fitting algorithms.
- nn_training: folder containing scripts for training the RNN and Transformer models.

## How to run the code

To run the code, you need to have Python 3.10.4 and PyTorch 1.11.0 installed. The datasets and checkpoints can be downloaded from https://doi.org/10.5281/zenodo.10998285.

The scripts that should be run are:

- `python -m nn_training.train_rnn` to train the RNN model.
- `python -m nn_training.train_transformer` to train the Transformer model.

If you try to run the scripts directly from their folders (e.g., python nn_training/train_rnn.py), you may get an error about an attempted relative import with no known parent package.

## References

- Original paper: [https://arxiv.org/abs/2211.04890v1](https://arxiv.org/abs/2211.04890v1)
- PyTorch: https://pytorch.org/
- Datasets and checkpoints: [https://zenodo.org/](https://doi.org/10.5281/zenodo.10998285)


