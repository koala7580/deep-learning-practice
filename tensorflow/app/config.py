"""Application run config definition.

"""
from collections import namedtuple as _namedtuple

RunConfig = _namedtuple('RunConfig',[
    'data_dir', 'model_dir', 'train_epochs',
    'epochs_between_evaluation', 'batch_size',
    'weight_decay', 'use_synthetic_data',
])

ResNetRunConfig = _namedtuple('ResNetRunConfig', RunConfig._fields + (
    'resnet_size', 'resnet_version'
))
