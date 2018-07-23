#!/bin/sh
ppath=$PYTHONPATH:$PWD
PYTHONPATH=$ppath python $PWD/scripts/cifar100_download_and_convert.py $@
