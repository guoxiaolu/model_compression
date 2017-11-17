import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
import numpy as np
from keras.optimizers import SGD
from keras.applications import resnet50, vgg16, imagenet_utils
from keras.preprocessing import image
from keras.models import load_model
from resnet_101 import resnet101_model, Scale
import cv2
from imagenet_tool import id_to_synset, synset_to_id

val_gt = '/media/wac/backup/imagenet/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
val_file = '/media/wac/backup/imagenet/ILSVRC/ImageSets/CLS-LOC/val.txt'
val_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/val_dir'
categories = './categories.txt'

f = open(categories, mode='r')
classes = [line.strip() for line in f.readlines()]

f_file = open(val_file, mode='r')
f_gt = open(val_gt, mode='r')
lines_file = f_file.readlines()
lines_gt = f_gt.readlines()

fname = [line.strip().split(' ')[0] for line in lines_file]
gt = [int(line.strip())for line in lines_gt]
val_dict = dict(zip(fname, gt))
#
#
# weights_path = './model/weights.00298.hdf5'
# # Test pretrained model
# # model = resnet101_model(weights_path)
# model = load_model(weights_path, custom_objects={'Scale':Scale})
# # model.summary()
# # model = VGG16(weights_path)
# sgd = SGD(lr=0.01, momentum=0.9, decay=4e-6, nesterov=False)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
# def processing_function(x):
#     # Remove zero-center by mean pixel, BGR mode
#     x[:, :, 0] -= 103.939
#     x[:, :, 1] -= 116.779
#     x[:, :, 2] -= 123.68
#     return x

# gen = image.ImageDataGenerator(preprocessing_function=processing_function)
#
# val_generator = gen.flow_from_directory(val_path, target_size=(224,224), classes=classes, shuffle=False,
#                                               batch_size=1)
# score = model.evaluate_generator(val_generator, val_generator.n)
# print score






top1 = 0
top5 = 0
for i,name in enumerate(fname):
    model = resnet50.ResNet50(weights='imagenet')

    img_path = os.path.join(val_path, name)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    result = imagenet_utils.decode_predictions(preds, top=5)[0]
    encode_result = [v[0] for v in result]
    if encode_result == gt[i]:
        top1 += 1
    if gt[i] in encode_result:
        top5 += 1
    print i, top1, top5