import os
import shutil
from imagenet_tool import id_to_synset

val_gt = '/media/wac/backup/imagenet/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
val_file = '/media/wac/backup/imagenet/ILSVRC/ImageSets/CLS-LOC/val.txt'
val_path = '/media/wac/backup/imagenet/ILSVRC2012_img_val'
dst_val_path = '/media/wac/backup/imagenet/val_dir_nobl'
blacklist = '/media/wac/backup/imagenet/devkit/data/ILSVRC2015_clsloc_validation_blacklist.txt'

f_file = open(val_file, mode='r')
f_gt = open(val_gt, mode='r')
f_bl = open(blacklist, mode='r')
lines_file = f_file.readlines()
lines_gt = f_gt.readlines()
lines_bl = f_bl.readlines()
f_file.close()
f_gt.close()
f_bl.close()

fname = [line.strip().split(' ')[0] for line in lines_file]
gt = [int(line.strip())for line in lines_gt]
bl = [int(line.strip()) for line in lines_bl]

for i, name in enumerate(fname):
    if i in bl:
        continue
    synset = id_to_synset(gt[i])
    dst_path = os.path.join(dst_val_path, synset)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    shutil.copy(os.path.join(val_path, name), os.path.join(dst_path, name))
    print name, gt[i], synset