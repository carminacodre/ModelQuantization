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

Additionally, it requires tensorflow and keras, but these
are installed together with the module.

Pay attention that the tensorlow version should be at least 1.7.
This can be checked with:

`
pip freeze
`

Aditionally, a requirements.txt is provided with the list of libraries used 
and needed.

If you need TensorFlow with GPU support uninstall the version which is installed together with pip 
and use the following command:

`
pip install tensorflow-gpu==1.7
`

## Usage

A protobuf file (having '.pb' extension) uses a serialization mechanism for 
structured data. It is platform independent and language neutral and thus is widely used
for serializing deep learning models both at the level of their graph definition and their 
parameters after training. 

However, after training a model, in order to bring it into production you usually want to
reduce its size or make it more suitable for its new environment. This can be done with the 
[Tensorflow- Graph Transform Tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md).

The functionalities exposed in this framework are an abstraction of the Graph Transform Tool and 
the main featues are:
* predefined lists of transformations to be applied for various scenarios
* functionality which takes a protobuf file, applies transormations and saves the
new model to a specified protobuf file
* utility function for summarization of graphs
* utility function for saving a model defined with
[Keras API](https://keras.io/) into a protobuf file

### quantization.operations module
The operations module defines three lists of transforms that can be used
on a model that needs to shift to production.

```
optimizing_for_deployment_trans
```

This prepares a model for deployment on a server or mobile device.
These transforms remove nodes not used during inference,
shrink constant expressions and optimize batch normalization. 

```
fix_missing_kernel_errors_trans
```

This one fixes the errors that might appear due to the fact that on mobile
the operations allowed are not the same as those used during training.
Such an error is `No OpKernel was registered to support Op errors`

```
optimize_quantize_weights_trans
```

This is a list of transformations which adds to the optimizing_for_deployment_trans
the weight quantization. Weight quantization is a mechanism of saving weights by using their range
and offset and reduces thus the memory needed.

The size of the module will be reduce. 

The three lists of operations listed above use as input the shape
"1,299,299,3".

If you want to apply the transforms on models with different input shape,
you can use one of the functions:
```
get_optimizing_for_deployment_for_shape(shape)
```
```
get_fix_missing_kernel_errors_for_shape(shape)
```
```
get_optimize_quantize_weights_for_shape(shape)
```

Example:
```
resized_trans = get_optimizing_for_deployment_for_shape('"1,300,300,3"')
```

The call above will return the following list:
```python
['strip_unused_nodes(type=float, shape="1,300,300,3")', 'remove_nodes(op=Identity, op=CheckNumerics)', 'fold_constants(ignore_errors=true)', 'fold_batch_norms', 'fold_old_batch_norms']
```

You can also define your own combination of operations to be applied.

### quantization.transform module

The transform module provides an entry point for the graph transform, more exactly the
function:
```
transform_graph(path_to_model, save_dir, file_name, inputs, outputs, transforms)
```

The parameters needed are:
* `path_to_model`: path to the '.pb' model you want to transform
* `save_dir` directory where the new model will be saved
* `file_name` file name of the new model after the transformations are applied
* `inputs` list of input tensors
* `outputs` list of output tensors
* `transforms` list of transforms to be applied. Note that the order is relevant.

Example of transforming a model with predefined transforms:
```
from quantization.transform import transform_graph
from quantization.operations import optimize_quantize_weights_trans 

transform_graph(path_to_model="models/ssd_inception_v2_coco.pb",
                save_dir="models",
                file_name ="ssd_inception_v2_coco_transformed_2.pb",
                inputs="image_tensor:0",
                outputs=['detection_boxes:0', 'detection_scores:0','detection_classes:0', 'num_detections:0'],
                transforms=optimize_quantize_weights_trans)
```

Example of transforming a model with custom transforms:
```
from quantization.transform import transform_graph

custom_trans = ['strip_unused_nodes(type=float, shape="1,299,299,3")',
                                       'fold_constants(ignore_errors=true)',
                                       'fold_batch_norms']
transform_graph(path_to_model="models/ssd_inception_v2_coco.pb",
                save_dir="models",
                file_name="ssd_inception_v2_coco_transformed_custom.pb",
                inputs="image_tensor:0",
                outputs=['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'],
                transforms=custom_trans)
```

### quantization.utils module

In this module you can find the utility function which saves a Keras model
to a protobuf file.

Note that tensorflow backend must be used for Keras.

```
save_keras_model_to_pb(save_dir, file_name, model, nr_outputs = 1, output_node_prefix= "output_node")
```

The parameters needed are:
* `save_dir`: directory where the new model will be saved
* `file_name`: file name of the new model after the transformations are applied
* `model`: keras Model
* `outputs`: number of outputs from the model
* `output_node_prefix`: OPTIONAL. prefix to be added to the output node names

Example of creating a simple model its conversion:
```

from keras.layers import Dense
from keras.models import Sequential
from quantization.utils import save_keras_model_to_pb

def create_test_model():
    #create basic model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model
    
model = create_test_model()
save_keras_model_to_pb(save_dir="models", file_name="test.pb", model=model)

```

Additionally, in this module there is a functionality for printing the graph as a summary.
```
summarize_graph(path_to_model)
```
The `path_to_model` parameter represents the path to the Protobuf file
which you want to analyze.

Example of summarizing a graph:
```
from quantization.utils import summarize_graph

summarize_graph("models/ssd_inception_v2_coco.pb")
```

## Testing

Tests and more examples can be found here:
https://github.com/carminacodre/TestModelQuantization



