import cv2
from keras.models import model_from_json
from keras.layers import Convolution2D, Dense, BatchNormalization, Merge
from keras.optimizers import SGD
import keras.backend as K
import numpy as np
import json
from resnet_101 import resnet101_model, Scale
from keras.applications import vgg16, resnet50, inception_v3
from keras.utils import plot_model
nb_classes = 1000
compression_ratio = 0.4

im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)

# Remove train image mean
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68

im = np.expand_dims(im, axis=0)
label = np.zeros((1, nb_classes))
label[0, 281] = 1

if K.image_dim_ordering() == 'th':
    channels_idx = 1
else:
    channels_idx = -1


model = resnet101_model('/Users/Lavector/model/keras_models/resnet101_weights_tf.h5')
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# model = inception_v3.InceptionV3()
# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)


def get_layer_output(layer, x):
    layer_function = K.function([layer.input, K.learning_phase()], [layer.output])
    out = layer_function([x, 0])[0]
    return out

# keras/issues/2226
def get_gradients(model, x):
    # Get gradient tensors
    trainable_weights = model.trainable_weights  # weight tensors
    weights = []
    layers_name = []
    for weight in trainable_weights:
        if model.get_layer(weight.name.split('/')[0]).trainable:
            weights.append(weight)
            layers_name.append(weight.name[:-2])

    gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors


    input_tensors = [model.inputs[0],  # input data
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    out = get_gradients(x)
    return dict(zip(layers_name, out))

def get_filtered_idx(name, filter_num, gradient):
    gradient_abs = np.abs(gradient)
    gradient_sum = np.sum(np.sum(np.sum(gradient_abs, axis=0), axis=0), axis=0)
    sorted_idx = np.argsort(gradient_sum)

    filtered_idx = sorted_idx[int(filter_num * compression_ratio):]

    return filtered_idx

def get_input_layer_name(layer):
    input_layers = None
    nodes = layer.inbound_nodes
    if len(nodes) == 1:
        node = nodes[0]
        input_layers = node.inbound_layers
    return input_layers

def get_last_conv_layer_name(layer):
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

# layers = model.layers
# hubs = {}
# for i, layer in enumerate(layers):
#     name = layer.name
#     input_layers = get_input_layer_name(layer)
#     if len(input_layers) > 1:
#         print name
#         input_conv_layers = []
#         for input_layer in input_layers:
#             input_conv_layer_name = get_last_conv_layer_name(input_layer)
#             input_conv_layers.append(input_conv_layer_name)
#         hubs[name] = input_conv_layers
#         print hubs[name]




layers = model.layers
x = im
y = label
gradients = get_gradients(model, [x, np.ones(y.shape[0]), y, 0])
conv_filtered_idx = {}
print 'sort convolutional layer gradient'
# sort convolution2D gradient
model_json = model.to_json()
model_structure = json.loads(model_json)

model_class_name = model_structure['class_name']
if model_class_name == 'Model':
    model_layer_name = [layer['name'] for layer in model_structure['config']['layers']]
for i, layer in enumerate(layers):
    name = layer.name
    print name
    # x = get_layer_output(layer, x)
    if isinstance(layer, Convolution2D):
        filter_num = layer.filters

        filtered_idx = get_filtered_idx(name, filter_num, gradients[name+'/kernel'])
        conv_filtered_idx[name] = filtered_idx
        if model_class_name == 'Model':
            idx = model_layer_name.index(name)
            model_structure['config']['layers'][idx]['config']['filters'] = len(conv_filtered_idx[name])
        elif model_class_name == 'Sequential':
            model_structure['config'][i]['config']['filters'] = len(conv_filtered_idx[name])
        else:
            pass
        print filter_num, filtered_idx
    elif isinstance(layer, Dense):
        pass

model_json = json.dumps(model_structure)
new_model = model_from_json(model_json, custom_objects={'Scale':Scale})
new_model.summary()

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
            input_layer_name = get_last_conv_layer_name(layer)

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

