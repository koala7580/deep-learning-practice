#!/bin/sh
ppath=$PYTHONPATH:$PWD
PYTHONPATH=$ppath python $PWD/scripts/cifar10_download_and_convert.py $@
