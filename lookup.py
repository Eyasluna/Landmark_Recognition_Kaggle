# coding: utf-8
# Author: Yibo Fu
# 6/2/2018
import pickle
import os
import cv2
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor

def lookup_file(filename):
    global images_in_leaves_table
    if os.path.isfile('./features-leafs/%s.cache' % filename):
        return None
    print('start %s' % filename)
    try:
        with open('./features/%s.cache' % filename, 'rb') as f:
            des = pickle.load(f)
            print('done load %s' % filename)
            out = []
            for d in des:
                print('look up %s' % d)
                leafID = lookup(d, 0)
                print('done')
                out.append(leafID)
            with open('./features-leafs/%s.cache' % filename, 'wb') as of:
                pickle.dump(out, of)
            print('done on %s' % filename)
    except:
        print('fail on %s' % filename)
        pass
    return None


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
    with ThreadPoolExecutor(max_workers=40) as pool:
        fs = map(lambda file: file[:len(file) - 6], os.listdir('./features'))
        for f in fs:
            pool.submit(lookup_file, f)
