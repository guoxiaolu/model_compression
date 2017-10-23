import os
import shutil
from random import sample

src_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/train'
dst_path = '/media/wac/backup/imagenet/ILSVRC/Data/CLS-LOC/train_sample1'
sample_ratio = 0.2

dirs = os.listdir(src_path)
for onedir in dirs:
    all = os.listdir(os.path.join(src_path, onedir))
    sample_all = sample(all, int(len(all) * sample_ratio))
    for one_sample in sample_all:
        src_abs_path = os.path.join(os.path.join(src_path, onedir), one_sample)
        dir_path = os.path.join(dst_path, onedir)
        dst_abs_path = os.path.join(dir_path, one_sample)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        shutil.copy(src_abs_path, dst_abs_path)