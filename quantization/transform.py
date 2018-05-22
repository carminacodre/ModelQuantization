import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph


def transform_graph(path_to_model, save_dir, file_name, inputs, outputs, transforms):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')

            transformed_graph_def = TransformGraph(graph_def,
                                                   inputs=inputs,
                                                   outputs=outputs,
                                                   transforms=transforms)

            graph_io.write_graph(graph_or_graph_def=transformed_graph_def, name=file_name, as_text=False,
                                 logdir=save_dir)
            print("Model after transforms"
                  + str(transforms) +
                  " has been save to "
                  + str(os.path.join(save_dir,file_name)))
