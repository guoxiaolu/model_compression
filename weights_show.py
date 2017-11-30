from keras.applications import resnet50
from keras.optimizers import SGD
from keras.layers import Convolution2D
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

model = resnet50.ResNet50()
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

total_kernels = 0
pruned_kernels = 0
weights = {}
for layer in model.layers:
    layer_weight = layer.get_weights()
    if isinstance(layer, Convolution2D):
        weights[layer.name + '/kernel'] = layer_weight[0]
        weights[layer.name + '/bias'] = layer_weight[1]

        weight_abs = np.abs(layer_weight[0])
        weight_sum = np.sum(np.sum(np.sum(weight_abs, axis=0), axis=0), axis=0)

        # print weight_sum
        mean = np.mean(weight_sum)
        std = np.std(weight_sum)
        # plt.bar(range(0, weight_sum.shape[0]), weight_sum.flatten())
        data = weight_sum.flatten()
        data_iqr = data[(data < mean - std)]
        data_rest = data[(data >= mean - std)]
        total_kernels += data.shape[0]
        pruned_kernels += data_iqr.shape[0]
        print data.shape[0], data_iqr.shape[0]
        plt.boxplot(data, showmeans=True, sym='*')
        plt.scatter(np.ones_like(data_iqr), data_iqr, marker='o')
        plt.scatter(np.ones_like(data_rest), data_rest, marker='x')
        plt.title("w values after 'imagenet' weights are loaded")
        plt.show()
print total_kernels, pruned_kernels