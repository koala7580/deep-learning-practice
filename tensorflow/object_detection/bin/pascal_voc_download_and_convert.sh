#!/bin/sh
ppath=$PYTHONPATH:$PWD
PYTHONPATH=$ppath python $PWD/scripts/pascal_voc_download_and_convert.py $@
