#!/bin/sh
CIFAR_FILENAME='cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL="https://www.cs.toronto.edu/~kriz/$CIFAR_FILENAME"
DATA_DIR='/tmp/cifar10_data'

mkdir -p $DATA_DIR

OLD_PWD=$PWD
cd $DATA_DIR
wget $CIFAR_DOWNLOAD_URL
tar -xvzf $CIFAR_FILENAME
cd $OLD_PWD
