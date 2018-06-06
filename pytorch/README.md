## Environment Prepare

Install `pyenv` and `pyenv-virtualenv`:

    $ brew install pyenv pyenv-virtualenv

Add init script to .bashrc or .zshrc:

    $ eval "$(pyenv init -)"
    $ eval "$(pyenv virtualenv-init -)"

Install anaconda3:

    $ pyenv install anaconda3-x.x.x

Activate anaconda environment:

    $ pyenv activate anacodna3-x.x.x

Create virtual environment:

    $ pyenv virtualenv pytorch

Activate virutalenv

    $ conda activate pytorch

Install anaconda meta package:

    $ conda install anaconda

Install tensorflow

    $ conda install pytorch torchvision -c pytorch 

