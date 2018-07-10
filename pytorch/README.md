## Environment Prepare

### Linux

Download the miniconda:

    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

### macOS

Install with Homebrew

    $ brew cask install miniconda

### Then

Add init script to .bashrc or .zshrc:

    . /prefix/to/miniconda3/etc/profile.d/conda.sh

Create virtual environment:

    $ conda create -n pytorch python=3.6 anaconda

Activate virutalenv

    $ conda activate pytorch

Install PyTorch:

    $ conda install pytorch torchvision -c pytorch 

or for linux with CUDA 9.0 which is used by tensorflow:

    $ conda install pytorch torchvision cuda90 -c pytorch

