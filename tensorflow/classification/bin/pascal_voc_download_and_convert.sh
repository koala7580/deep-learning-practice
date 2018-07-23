#!/bin/sh
ppath=$PYTHONPATH:$PWD
PYTHONPATH=$ppath python $PWD/practice/scripts/pascal_voc_download_and_convert.py $@
