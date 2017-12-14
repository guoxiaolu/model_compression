import numpy as np
import glob

number = 208
npy_list = glob.glob('./npy/*.npy')


for i, npy in enumerate(npy_list):
    gradient = np.load(npy).item()
    if i == 0:
        gradient_all = gradient
    else:
        for k,v in gradient_all.iteritems():
            gradient_all[k] = v + gradient[k]

for k,v in gradient_all.iteritems():
    gradient_all[k] = v / number

np.save('all', gradient_all)