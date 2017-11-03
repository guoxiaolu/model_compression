import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.models import load_model
from keras.layers import Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras.backend as K
from keras.optimizers import SGD
from resnet_101 import Scale

categories = './categories.txt'
train_img_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/train'
val_img_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/val_dir'

f = open(categories, mode='r')
classes = [line.strip() for line in f.readlines()]
nb_classes = len(classes)

compression_ratio = 0.4
image_size = (224, 224)
batch_size = 32
nb_epoch = 100

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

# datagenerator
gen = ImageDataGenerator(preprocessing_function=processing_function)
train_generator = gen.flow_from_directory(train_img_path, target_size=image_size, classes=classes, shuffle=True,
                                              batch_size=batch_size)
val_generator = gen.flow_from_directory(val_img_path, target_size=image_size, classes=classes, shuffle=False,
                                              batch_size=1)

# model_pruning = load_model('./resnet101_weights_tf_pruning.h5', custom_objects={'Scale':Scale})
model_pruning = load_model('./model/weights.00017.hdf5', custom_objects={'Scale':Scale})
# model_pruning.summary()

layers = model_pruning.layers
for layer in layers:
    if isinstance(layer, Convolution2D):
        layer.trainable = False

sgd = SGD(lr=0.1, momentum=0.9, decay=4e-6, nesterov=False)
model_pruning.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./model/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model_pruning.fit_generator(train_generator, train_generator.n/batch_size, nb_epoch=nb_epoch, callbacks=[tb, mc], validation_data=val_generator, validation_steps=val_generator.n, initial_epoch=18)

model_pruning.save('./resnet101_weights_tf_pruning_retrain.h5')