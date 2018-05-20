import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def transform_graph(model_path, save_to, inputs, outputs, transforms):
    print("Transforming model")