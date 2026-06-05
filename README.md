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

The code was developed using Python 3.10.4 and PyTorch 1.11.0. Set up the python environment by anaconda.

To install anaconda, please check [Linux installer](https://www.anaconda.com/docs/getting-started/anaconda/install/linux-install#terminal) for more information.

With conda, install Python 3.10.4 by:
- ```
  conda create --name py3104_env -c anaconda python=3.10.4
  ```

Activate this envrionment by:
- ```
  conda activate py3104_env
  ```

Install PyTorch 1.11.0 with pip:
- ```
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

Install numpy with pip:
- ```
  pip install numpy==1.26.4
  ```

Inside the `trajectory_fitting-main` directory, create a folder to hold the training outputs:
- ```
  mkdir models
  ```

The datasets can be found on nnhome storage space here: `/storage/shared/chiaki/superFGD/training`.

## How to run the code

### Interactive jobs

To run the code interactively, first is to log into a computation node with GPU: 
- ```
  srun --gres=gpu --x11 --pty bash -i
  ```
On the node always activate the right pythonic environment before trainning:
- ```
  conda activate py3104_env
  ```

To aviod errors about package import, the scripts should be run inside the `trajectory_fitting-main` folder:
- `cd INSTALLL_DIR/trajectory_fitting-main` to change directory to the correct one.
- `python -m nn_training.train_rnn` to train the RNN model.
- `python -m nn_training.train_transformer` to train the Transformer model.

If you try to run the scripts directly from their folders (e.g., python nn_training/train_rnn.py), you may get an error about an attempted relative import with no known parent package.

### Remote jobs

To run the code with remote jobs, change directory to `jobs` and make necessary modifications to a `.sh` script according to user demands. 

Submit the shell script by:
- ```
  sbatch run_train_trans_model.sh 
  ```

## Trainning setup

The common training settings are stored in `trajectory_fitting-main/modules/constants.py`. 

Here are a few key variables in the set up:
- `TRAINING_DATASET` the directory to the dataset;
- `NUM_EPOCH` the number of total epochs in training;
- `BATCH_SIZE` the size for one batch of input samples.
   
## Changes for GPU memory optimization

The majority of code is the same as the original repository, but with specific minor modifications to optimize the GPU memory comsupation for deployment on nnhome clusters.
Most of the changes are made for the training of a transformer model since it comsumes more memory compared to training a RNN model.

The key optimizations include:
- Use mixed precision of tensors with `torch.cuda.amp.autocast()` function, for example, [nn_training/train_transformer.py #L53](https://github.com/mojiastonybrook/SuperFGDTrackML/blob/0fb7b84400ea36ea32b53b91001a911f35fa3620/trajectory_fitting-main/nn_training/train_transformer.py#L53)
  ```
  def train_epoch(model, optim, disable_tqdm):
      ...
  
      with torch.cuda.amp.autocast():
          for i, data in t:
              ...

      return ...
  ```
- Reduce batchsize, [modules/constants.py #L37](https://github.com/mojiastonybrook/SuperFGDTrackML/blob/0fb7b84400ea36ea32b53b91001a911f35fa3620/trajectory_fitting-main/modules/constants.py#L37)
  `BATCH_SIZE = 16`
- Reduce number of parallel  workers in dataloader, for example [nn_training/train_transformer.py #L135](https://github.com/mojiastonybrook/SuperFGDTrackML/blob/0fb7b84400ea36ea32b53b91001a911f35fa3620/trajectory_fitting-main/nn_training/train_transformer.py#L135)
  ```
  train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=BATCH_SIZE,
                          num_workers=1, shuffle=True)
  ```
- Set PyTorch memory allocator option by
  `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32`.

  It could be done in a job script as [https://github.com/mojiastonybrook/SuperFGDTrackML/blob/0fb7b84400ea36ea32b53b91001a911f35fa3620/jobs/run_train_trans_model.sh#L10](https://github.com/mojiastonybrook/SuperFGDTrackML/blob/0fb7b84400ea36ea32b53b91001a911f35fa3620/jobs/run_train_trans_model.sh#L10)
