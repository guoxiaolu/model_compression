import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
from keras.models import model_from_json
from keras.layers import Convolution2D, Dense, BatchNormalization, merge
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
import json
from random import sample
from resnet_101 import resnet101_model, Scale
from keras.applications import vgg16, resnet50, inception_v3
from keras.utils import plot_model

categories = './categories.txt'
train_img_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/train_sample1'

f = open(categories, mode='r')
classes = [line.strip() for line in f.readlines()]
nb_classes = len(classes)

compression_ratio = 0.4
image_size = (224, 224)
batch_size = 16

def processing_function(x):
    # Remove zero-center by mean pixel, BGR mode
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

if K.image_dim_ordering() == 'th':
    channels_idx = 1
else:
    channels_idx = -1


# model = inception_v3.InceptionV3()
# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)


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

def get_gradients(model, x, gradients_all, layers_name):
    '''
    This func is based on #keras/issues/2226.
    :param model: the model instance
    :param x: input, like [x, np.ones(y.shape[0]), y, 0] : x is np.array(None, height, width, channel),
              y is one-hot (1, nb_classes)
    :return: dict of weights
    '''

    input_tensors = [model.inputs[0],  # input data
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients_all)
    out = get_gradients(x)
    return dict(zip(layers_name, out))

def get_filtered_idx(filter_num, gradient):
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
    sorted_idx = np.argsort(gradient_sum)
    filtered_idx = sorted_idx[int(filter_num * compression_ratio):]
    return filtered_idx.tolist()

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

if __name__ == '__main__':

    model = resnet101_model('./model/resnet101_weights_tf.h5')
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # datagenerator
    gen = ImageDataGenerator(preprocessing_function=processing_function)
    train_generator = gen.flow_from_directory(train_img_path, target_size=image_size, classes=classes, shuffle=True,
                                              batch_size=batch_size)

    # Get gradient tensors
    trainable_weights = model.trainable_weights  # weight tensors
    weights = []
    layers_name = []
    # weight name is different from layer name, as the weight is consisted of kernel and bias
    for weight in trainable_weights:
        if model.get_layer(weight.name.split('/')[0]).trainable:
            weights.append(weight)
            layers_name.append(weight.name[:-2])

    gradients_all = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors
    total = 0
    gradients = dict(zip(layers_name, [0]*len(layers_name)))
    for x_batch, y_batch in train_generator:
        gradient = get_gradients(model, [x_batch, np.ones(y_batch.shape[0]), y_batch, 0], gradients_all, layers_name)
        for k,v in gradient.iteritems():
            gradients[k] = v + gradients[k] * total
            gradients[k] = gradients[k] / (total + x_batch.shape[0])
        total += x_batch.shape[0]
        print total
        if total >= train_generator.n:
            break


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
    for i, layer in enumerate(layers):
        name = layer.name
        print name
        if isinstance(layer, Convolution2D):
            filter_num = layer.filters

            filtered_idx = get_filtered_idx(filter_num, gradients[name+'/kernel'])
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
    new_model = model_from_json(model_json, custom_objects={'Scale':Scale})
    # new_model.summary()


    # pruning filters
    print 'start pruning'
    input_layer_name = None
    for i, layer in enumerate(layers):
        name = layer.name
        print name
        weight = layer.get_weights()
        if isinstance(layer, Convolution2D):
            new_weight = []
            channels = layer.input_shape[channels_idx]

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
                    # select random idx
                    input_filters_num = new_model.layers[i].input_shape[-1]
                    input_filtered_idx = sample(input_filtered_idx, input_filters_num)

            else:
                if input_layer_name in conv_filtered_idx:
                    input_filtered_idx = conv_filtered_idx[input_layer_name]
                else:
                    input_filtered_idx = range(0, channels)

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
        else:
            pass
            # new_model.layers[i].set_weights(weight)

    new_model.save('resnet101_weights_tf_pruning.h5')

