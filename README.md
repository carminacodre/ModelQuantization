# ModelQuantization

This module facilitates the quantization of tensorflow models. 
The implementation is based on the guidelines from  
[Tensorflow- Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md).


The functionality exposed through this module allows you to use the
Graph Transform Tool directly from a python script and does not need 
building the tensorflow library with bazel.

## Installation

The installation can be done directly through pip.

`
pip install git+https://github.com/carminacodre/ModelQuantization
`

Note that the module is written in python 3.6.5

It additionally requires tensorflow and keras, but these
are installed together with the module.

## Usage
