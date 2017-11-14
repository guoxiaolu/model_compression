from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import psutil
import timeit
import gc

def get_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info()


def build():
    model = Sequential()
    model.add(Dense(output_dim=4096, input_dim=4096, init="glorot_uniform"))
    model.add(Activation("relu"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


if __name__ == '__main__':
    for i in xrange(100):
        gc.collect()
        t = timeit.timeit('build()', number=1, setup="from __main__ import build")
        mem = get_mem_usage()
        print('build time: {}, mem: {}'.format(t, mem))