import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import model_from_json
from keras.layers import Convolution2D
import keras.backend as K
import json
from keras.applications import vgg16, resnet50
from keras.utils import plot_model
from inner_func import get_hubs_last_conv_name, get_filtered_idx, get_last_conv_layer_name, recursive_find_root_conv

# compression_ratio = 0.2
std_times = 0.8

if K.image_dim_ordering() == 'th':
    channels_idx = 1
else:
    channels_idx = -1

def pruning_layer(layer, model_class_name, hubs, conv_filtered_idx, input_filters_num):
    channels = layer.input_shape[channels_idx]
    input_layer_name = None
    if model_class_name == 'Model':
        layer_name = get_last_conv_layer_name(layer)
        # find last convolutional layer
        if layer_name not in hubs:
            input_layer_name = layer_name

            if input_layer_name in conv_filtered_idx:
                input_filtered_idx = conv_filtered_idx[input_layer_name]
            else:
                input_filtered_idx = range(0, channels)
        # find merge/concat layer
        else:
            input_layer_name = recursive_find_root_conv(hubs[layer_name], [], hubs)
            input_filtered_idx = []
            for conv_layer_name in input_layer_name:
                if conv_layer_name in conv_filtered_idx:
                    input_filtered_idx += conv_filtered_idx[conv_layer_name]
                else:
                    input_filtered_idx += range(0, channels)
            input_filtered_idx = list(set(input_filtered_idx))
            input_filtered_idx = input_filtered_idx[:input_filters_num]

    else:
        if input_layer_name in conv_filtered_idx:
            input_filtered_idx = conv_filtered_idx[input_layer_name]
        else:
            input_filtered_idx = range(0, channels)

    return input_filtered_idx

if __name__ == '__main__':
    model = resnet50.ResNet50()
    model.summary()

    weights = {}
    for layer in model.layers:
        layer_weight = layer.get_weights()
        if isinstance(layer, Convolution2D):
            weights[layer.name+'/kernel'] = layer_weight[0]
            weights[layer.name + '/bias'] = layer_weight[1]

    # get hubs (layers like merge, concatenate, which has many inputs) last convolution layer name
    print 'get hubs'
    layers = model.layers
    hubs = get_hubs_last_conv_name(layers)

    conv_filtered_idx = {}
    hubs_filtered_idx = {}
    print 'sort convolutional layer gradient and reconstruct model'
    # sort convolution2D gradient
    model_json = model.to_json()
    model_structure = json.loads(model_json)

    model_class_name = model_structure['class_name']
    if model_class_name == 'Model':
        model_layer_name = [layer['name'] for layer in model_structure['config']['layers']]

    # special convolutional layer (resnet short-cut), don't prune
    special_conv = []
    for conv in hubs.values():
        special_conv += conv

    # sort based on each layer
    for i, layer in enumerate(layers):
        name = layer.name
        print name
        if isinstance(layer, Convolution2D) and name not in special_conv:
            filter_num = layer.filters

            filtered_idx = get_filtered_idx(filter_num, weights[name+'/kernel'], std_times)
            conv_filtered_idx[name] = filtered_idx
            if model_class_name == 'Model':
                idx = model_layer_name.index(name)
                model_structure['config']['layers'][idx]['config']['filters'] = len(conv_filtered_idx[name])
            elif model_class_name == 'Sequential':
                model_structure['config'][i]['config']['filters'] = len(conv_filtered_idx[name])
            else:
                pass
            print filter_num, filtered_idx
        else:
            pass

    model_json = json.dumps(model_structure)
    new_model = model_from_json(model_json)
    new_model.summary()
    plot_model(new_model, show_shapes=True)

    # pruning filters
    print 'start pruning'
    input_layer_name = None
    for i, layer in enumerate(layers):
        name = layer.name
        print name
        weight = layer.get_weights()
        if isinstance(layer, Convolution2D) and name not in special_conv:
            new_weight = []
            input_filters_num = new_model.layers[i].input_shape[-1]
            input_filtered_idx = pruning_layer(layer, model_class_name, hubs, conv_filtered_idx, input_filters_num)

            output_filtered_idx = conv_filtered_idx[name]
            new_weight_kernel = weight[0][:, :, input_filtered_idx, :]
            new_weight_kernel = new_weight_kernel[:, :, :, output_filtered_idx]
            new_weight.append(new_weight_kernel)

            if layer.bias != None:
                new_weight_bias = weight[1][output_filtered_idx]
                new_weight.append(new_weight_bias)

            new_model.layers[i].set_weights(new_weight)

            if model_class_name == 'Sequential':
                input_layer_name = name
        elif isinstance(layer, Convolution2D) and name in special_conv:
            new_weight = []
            input_filters_num = new_model.layers[i].input_shape[-1]
            input_filtered_idx = pruning_layer(layer, model_class_name, hubs, conv_filtered_idx, input_filters_num)

            new_weight_kernel = weight[0][:, :, input_filtered_idx, :]
            new_weight.append(new_weight_kernel)

            if layer.bias != None:
                new_weight_bias = weight[1][:]
                new_weight.append(new_weight_bias)

            new_model.layers[i].set_weights(new_weight)

            if model_class_name == 'Sequential':
                input_layer_name = name
        else:
            pass

    new_model.save('resnet50_weights_tf_pruning_%0.1fstd.h5'%(std_times))

