## Environment Prepare

Install `pyenv` and `pyenv-virtualenv`:

    $ brew install pyenv pyenv-virtualenv

Add init script to .bashrc or .zshrc:

    $ eval "$(pyenv init -)"
    $ eval "$(pyenv virtualenv-init -)"

Install anaconda3:

    $ pyenv install anaconda3-x.x.x

Create virtual environment:

    $ pyenv virtualenv anaconda3-x.x.x tensorflow

Activate virutalenv

    $ pyenv activate tensorflow
    $ pyenv local tensorflow

Install tensorflow

    $ conda install -c conda-forge tensorflow

