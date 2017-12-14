#!/usr/bin/env python2.7
from __future__ import division

# ---
# Keras implementation of testing ResNet-50 on ImageNet 2012 Val.
# Li Ding, Oct. 20, 2017
# ---
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16, decode_predictions
# from keras.applications.resnet50 import ResNet50, decode_predictions
from keras.models import load_model
import xmltodict
import Queue
import threading
from tqdm import tqdm
import glob
import cv2
from resnet_101 import Scale

# set path
data_path = '/media/wac/backup/imagenet/ILSVRC'
img_path = data_path + '/Data/CLS-LOC/val'
label_path = data_path + '/Annotations/CLS-LOC/val'


# set parameters
batch_size = 16
n = 50000  # number of images, 50,000 for val
graph = tf.get_default_graph()

# get ground truth labels
names = glob.glob(label_path+'/*')
names.sort()

def processing_function(x):

    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x

# threading for faster inference
def data_loader(q, ):
    for start in range(0, n, batch_size):
        x_batch = []
        end = min(start + batch_size, n)
        ids_test_batch = names[start:end]
        for idr in ids_test_batch:
            id = idr.split('/')[-1].split('.')[0]
            img = cv2.imread(img_path+'/{}.JPEG'.format(id))
            img = img[...,::-1]
            r = 256/min(img.shape[:2])
            img = cv2.resize(img, dsize=(0,0), fx=r, fy=r)
            x = int((img.shape[0]-224)/2)
            y = int((img.shape[1]-224)/2)
            img = img[x:x+224,y:y+224,:]
            img = img.astype(float)
            img = processing_function(img)
            x_batch.append(img)
        x_batch = np.array(x_batch)
        q.put(x_batch)

def predictor(q, ):
    for i in tqdm(range(0, n, batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds.append(model.predict_on_batch(x_batch))


# get prediction
if os.path.isfile(data_path + "/val_prob.csv"):
    preds = np.genfromtxt(data_path + "/val_prob.csv", delimiter=",")
else:
    # set model
    # model = ResNet50(weights='imagenet')
    model = load_model('./model/weights.00007.hdf5')

    preds = []
    q = Queue.Queue(maxsize=16)
    t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q,))
    t2 = threading.Thread(target=predictor, name='Predictor', args=(q,))
    print('Predicting on {} samples with batch_size = {}...'.format(n, batch_size))
    t1.start()
    t2.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()

    preds = np.array(preds).reshape(n,1000)
    np.savetxt(data_path + "/val_prob.csv", preds, '%.2f', delimiter=",")
    print 'Prediction saved!', data_path + "/val_prob.csv"


# get top-5 prediction
p = decode_predictions(preds, top=5)
# get top-1 prediction
# p = decode_predictions(preds, top=1)


gt = []
for i in names:
    with open(i) as f:
        a = xmltodict.parse(f.read())['annotation']['object']
        l = []
        if isinstance(a,list):
            for j in a:
                l.append(j['name'])
        else:
            l.append(a['name'])
        gt.append(list(set(l)))


# evaluate
err = []
for i,j in zip(gt,p):
    err.append(sum([h not in [k[0] for k in j] for h in i])/len(i))

err = np.array(err)
ind = np.arange(n)
wrong = ind[err==1]

print 'Top-5 Error:', sum(err)/len(err)
# print 'Top-1 Error:', sum(err)/len(err)


'''
# get true labels, instead of IDs
from nltk.corpus import wordnet
syns = list(wordnet.all_synsets())
offsets_list = [(s.offset(), s) for s in syns]
offsets_dict = dict(offsets_list)


# explore the wrong examples
from PIL import Image
i = int(np.random.random() * len(wrong))
Image.open(img_path+'/ILSVRC2012_val_{}.JPEG'.format('%08d'%(wrong[i]+1))).show()
print 'Ground Truth:', offsets_dict[int(gt[wrong[i]][0][1:])]
print 'Prediction: ', p[wrong[i]]
'''

