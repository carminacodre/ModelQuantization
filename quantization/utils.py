import os
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def summarize_graph(path_to_model):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

def save_keras_model_to_pb(save_dir, file_name, model, nr_outputs = 1, output_node_prefix= "output_node"):

    pred = [None] * nr_outputs
    pred_node_names = [None] * nr_outputs
    for i in range(nr_outputs):
        pred_node_names[i] = output_node_prefix + str(i)
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)

    #write the graph to ouput
    graph_io.write_graph(graph_or_graph_def=constant_graph, name= file_name, logdir=save_dir, as_text=False)
    print("Saved the keras model to " +
          str(os.path.join(save_dir,file_name)))