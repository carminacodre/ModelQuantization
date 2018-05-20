import tensorflow as tf

def summarize_graph(path_to_model):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as f:
            serialized_graph = f.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]