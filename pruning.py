from keras.models import load_model, model_from_json
from keras.layers import Convolution2D, Dense
import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import json

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
nb_classes = 10
compression_ratio = 0.4

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    channels_idx = 1
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    channels_idx = -1

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model('./weights_mnist_cnn.h5')
model.summary()


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
    gradient_abs = np.abs(gradients[name + '/kernel'])
    gradient_sum = np.sum(np.sum(np.sum(gradient_abs, axis=0), axis=0), axis=0)
    sorted_idx = np.argsort(gradient_sum)

    filtered_idx = sorted_idx[int(filter_num * compression_ratio):]

    return filtered_idx


layers = model.layers
x = X_train[0:1]
y = Y_train[0:1]
gradients = get_gradients(model, [x, np.ones(y.shape[0]), y, 0])
conv_filtered_idx = {}
print 'sort convolutional layer gradient'
# sort convolution2D gradient
model_json = model.to_json()
model_structure = json.loads(model_json)
for i, layer in enumerate(layers):
    name = layer.name
    print name
    # x = get_layer_output(layer, x)
    if isinstance(layer, Convolution2D):
        filter_num = layer.filters

        filtered_idx = get_filtered_idx(name, filter_num, gradients[name + '/kernel'])
        conv_filtered_idx[name] = filtered_idx
        model_structure['config'][i]['config']['filters'] = len(conv_filtered_idx[name])
        print filter_num, filtered_idx
    elif isinstance(layer, Dense):
        pass

model_json = json.dumps(model_structure)
new_model = model_from_json(model_json)
new_model.summary()

# pruning filters
print 'start pruning'
last_filtered_name = None
for i, layer in enumerate(layers):
    name = layer.name
    print name
    weight = layer.get_weights()
    if isinstance(layer, Convolution2D):
        new_weight = []
        channels = layer.input_shape[channels_idx]

        if last_filtered_name in conv_filtered_idx:
            input_filtered_idx = conv_filtered_idx[last_filtered_name]
        else:
            input_filtered_idx = range(0, channels)

        output_filtered_idx = conv_filtered_idx[name]
        new_weight_kernel = weight[0][:, :, input_filtered_idx, :]
        new_weight_kernel = new_weight_kernel[:, :, :, output_filtered_idx]
        new_weight_bias = weight[1][output_filtered_idx]
        new_weight.append(new_weight_kernel)
        new_weight.append(new_weight_bias)

        new_model.layers[i].set_weights(new_weight)

        last_filtered_name = name
    elif isinstance(layer, Dense):
        pass
        # units = layer.units
    else:
        new_model.layers[i].set_weights(weight)

new_model.save('weights_mnist_cnn_pruning.h5')
