# coding: utf-8
# Author: Yibo Fu
# 6/2/2018
import os
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor

from multiprocessing.pool import ThreadPool
import numpy as np

src = './train'
sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)

def extra_feature(file):
    _, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(src + "/" + file),
                                                cv2.COLOR_BGR2GRAY), None)
    print('extracted feature %s' % file)
    with open("./features/%s.cache" % file, 'wb') as f:
        pickle.dump(des, f)
    return des


def concat_q(qu, a, b):
    qu.append(a + b)


def dump_features():
    with ThreadPoolExecutor(max_workers=40) as executor:
        files = os.listdir(src)
        executor.map(extra_feature, files)
        # flat_features = []
        #
        # for sub in features:
        #     for d in sub:
        #         flat_features.append(d)

dump_features()
