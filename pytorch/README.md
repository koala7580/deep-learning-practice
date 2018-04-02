## Environment Prepare

Install `pyenv` and `pyenv-virtualenv`:

    $ brew install pyenv pyenv-virtualenv

Add init script to .bashrc or .zshrc:

    $ eval "$(pyenv init -)"
    $ eval "$(pyenv virtualenv-init -)"

Install anaconda3:

    $ pyenv install anaconda3-x.x.x

Create virtual environment:

    $ pyenv virtualenv anaconda3-x.x.x pytorch

Activate virutalenv

    $ pyenv activate pytorch
    $ pyenv local pytorch

Install anaconda meta package:

    $ conda install anaconda

Install tensorflow

    $ conda install pytorch torchvision -c pytorch 

