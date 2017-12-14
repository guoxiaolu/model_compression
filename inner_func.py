import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Convolution2D
import keras.backend as K
import numpy as np
import psutil
import os

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()

def processing_function(x):
    # Remove zero-center by mean pixel, BGR mode
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

def get_layer_output(layer, x):
    '''
    Get layer output based on its input.
    :param layer: model.layer
    :param x: layer input
    :return: the output of this layer
    '''
    layer_function = K.function([layer.input, K.learning_phase()], [layer.output])
    out = layer_function([x, 0])[0]
    return out

def get_gradients(model, x, layers_name):
    '''
    This func is based on #keras/issues/2226.
    :param model: the model instance
    :param x: input, like [x, np.ones(y.shape[0]), y, 0] : x is np.array(None, height, width, channel),
              y is one-hot (1, nb_classes)
    :return: dict of weights
    '''

    gradients_all = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)  # gradient tensors

    input_tensors = [model.inputs[0],  # input data
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients_all)
    out = get_gradients(x)
    return dict(zip(layers_name, out))

def get_filtered_idx(filter_num, gradient, std_times):
    '''
    Sort gradient of each layer.
    gradient shape is (filter_kernel_size0, filter_kernel_size1, input_filter_num, output_filter_num),
    1st, get abs_gradient
    2nd, get sum, the shape is (1, output_filter_num)
    then sort it.
    :param filter_num: the filter number of this layer
    :param gradient: the gradient of this layer
    :return: list of filter index to be filtered
    '''
    gradient_abs = np.abs(gradient)
    gradient_sum = np.sum(np.sum(np.sum(gradient_abs, axis=0), axis=0), axis=0)
    # sorted_idx = np.argsort(gradient_sum)
    # filtered_idx = sorted_idx[int(filter_num * compression_ratio):]
    mean = np.mean(gradient_sum)
    std = np.std(gradient_sum)
    filtered_idx = np.where(gradient_sum > (mean - std_times * std))[0]
    return filtered_idx.tolist()

def get_gradient_sum(gradient):
    '''
    Sort gradient of each layer.
    gradient shape is (filter_kernel_size0, filter_kernel_size1, input_filter_num, output_filter_num),
    1st, get abs_gradient
    2nd, get sum, the shape is (1, output_filter_num)
    then sort it.
    :param filter_num: the filter number of this layer
    :param gradient: the gradient of this layer
    :return: list of filter index to be filtered
    '''
    gradient_abs = np.abs(gradient)
    gradient_sum = np.sum(np.sum(np.sum(gradient_abs, axis=0), axis=0), axis=0)
    return gradient_sum.tolist()

def get_input_layer_name(layer):
    '''
    Get input of one layer
    :param layer: layer instance
    :return: input layers
    '''
    input_layers = None
    nodes = layer.inbound_nodes
    if len(nodes) == 1:
        node = nodes[0]
        input_layers = node.inbound_layers
    return input_layers

def get_last_conv_layer_name(layer):
    '''
    Get last convolution/merge/concat layer name
    :param layer: layer instance
    :return: the last convolutional layer name or Merge, Concat layer (which has many inputs)
    '''
    name = layer.name
    aim_layer = layer
    while name != '':
        input_layers = get_input_layer_name(aim_layer)
        if input_layers != None:
            if len(input_layers) == 1:
                if isinstance(input_layers[0], Convolution2D):
                    name = input_layers[0].name
                    break
                else:
                    aim_layer = input_layers[0]
                    name = aim_layer.name
            elif len(input_layers) > 1:
                name = aim_layer.name
                break
            else:
                name = ''
        else:
            name = ''
    return name

def get_hubs_last_conv_name(layers):
    '''
    Get hubs (Merge, Concat) last convolutional layer name
    :param layers: model.layers
    :return: the dict of hub, key is hub name, value is its input
    '''
    hubs = {}
    for i, layer in enumerate(layers):
        name = layer.name
        input_layers = get_input_layer_name(layer)
        if len(input_layers) > 1:
            input_conv_layers = []
            for input_layer in input_layers:
                input_conv_layer_name = get_last_conv_layer_name(input_layer)
                input_conv_layers.append(input_conv_layer_name)
            hubs[name] = input_conv_layers
    return hubs

def recursive_find_root_conv(hub_values, new_hub_values, hubs):
    '''
    Recursive function, find all convolutional layer name of hub (Merge, Concat)
    :param hub_values: hub
    :param new_hub_values: one hub input
    :param hubs: hub dict
    :return: one hub input
    '''
    for v in hub_values:
        if v not in hubs:
            new_hub_values.append(v)
        else:
            recursive_find_root_conv(hubs[v], new_hub_values, hubs)
    return new_hub_values