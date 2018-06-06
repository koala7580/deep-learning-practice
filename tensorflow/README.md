## Environment Prepare

Install `pyenv` and `pyenv-virtualenv`:

    $ brew install pyenv pyenv-virtualenv

Add init script to .bashrc or .zshrc:

    $ eval "$(pyenv init -)"
    $ eval "$(pyenv virtualenv-init -)"

Install anaconda3:

    $ pyenv install anaconda3-x.x.x

Activate the anaconda environment:

    $ pyenv activate anaconda3-x.x.x

Create virtual environment:

    $ pyenv virtualenv tensorflow

Activate virutalenv:

    $ conda activate tensorflow

Install anaconda meta package:

    $ conda install anaconda

Install tensorflow

    $ pip install tensorflow

