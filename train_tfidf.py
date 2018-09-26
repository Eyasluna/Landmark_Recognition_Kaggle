# coding: utf-8

import pickle
import os
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)  # SIFT Feature extractor model
TOTAL = 7500
src = './train'


def weight(leafID):
    return math.log1p(TOTAL / 1.0 * len(images_in_leaves_table[leafID]))


def tfidf(filename):
    global images_in_leaves_table
    try:
        with open('./features/%s.cache' % filename, 'rb') as f:
            des = pickle.load(f)
            for d in des:
                leafID = lookup(d, 0)
                if filename in images_in_leaves_table[leafID]:
                    images_in_leaves_table[leafID][filename] += 1
                else:
                    images_in_leaves_table[leafID][filename] = 1
    except:
        print('fail on %s' % filename)
        pass


def lookup(descriptor, node):
    D = float("inf")
    goto = None
    for child in tree[node]:
        dist = np.linalg.norm([nodes[child] - descriptor])
        if D > dist:
            D = dist
            goto = child
    if tree[goto] == []:
        return goto
    return lookup(descriptor, goto)


with open("model", 'rb') as f:
    node_index, nodes, tree, images_in_leaves_table, avgDepth, doc = pickle.load(f)

    count = 0
    # with ThreadPoolExecutor(max_workers=40) as executor:
    caches = os.listdir('./features')
    total = len(caches)
    for file in caches:
        tfidf(file[:len(file) - 6])
        count += 1
        print("%d/%d" % (count, total))
    # files = os.listdir(src)

    # for file in files:
    #     tfidf(file)
    #     count += 1
    #     print("%d/%d" % (count, len(files)))
    # for name in executor.map(tfidf, files):
    #     pass

    for leafID in images_in_leaves_table:
        w = weight(leafID)
        for img in images_in_leaves_table[leafID]:
            if img not in doc:
                doc[img] = {}
            doc[img][leafID] = w * (images_in_leaves_table[leafID][img])

    for img in doc:
        s = 0.0
        for leafID in doc[img]:
            s += doc[img][leafID]
        for leafID in doc[img]:
            doc[img][leafID] /= s
    with open("model_doc", 'wb') as out:
        pickle.dump((node_index, nodes, tree, images_in_leaves_table, avgDepth, doc), out)
