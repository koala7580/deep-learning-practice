#!/bin/sh
ppath=$PYTHONPATH:$PWD
PYTHONPATH=$ppath python $PWD/practice/scripts/kline_main.py $@
