from keras.models import load_model
from keras.layers import Convolution2D
import keras.backend as K
from keras.datasets import mnist
from keras.utils import np_utils
from time import time

batch_size = 128
nb_classes = 10
nb_epoch = 6

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
compression_ratio = 0.2

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
start = time()
score = model.evaluate(X_test, Y_test, verbose=0)
end = time()
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('cost time:', end-start)

model_pruning = load_model('./weights_mnist_cnn_pruning.h5')
model_pruning.summary()
# score = model_pruning.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


layers = model_pruning.layers
for layer in layers:
    if isinstance(layer, Convolution2D):
        layer.trainable = False

model_pruning.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model_pruning.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

start = time()
score = model_pruning.evaluate(X_test, Y_test, verbose=0)
end = time()
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('cost time:', end-start)
model_pruning.save('./weights_mnist_cnn_pruning_retrain.h5')