import math
import os
import numpy as np
import pickle
import cv2
import time

TEST_DIR = 'cross-test'
TOTAL = 7500
sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)  # SIFT Feature extractor model

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

all_train_files = os.listdir("./train/")

def getScores(q):
    print('get score', time.time())
    scores = {}
    curr = [float("inf")]
    currimg = [""]
    for fname in all_train_files:
        img = fname
        scores[img] = 0
        if img not in doc:
            continue
        for leafID in images_in_leaves_table:
            indoc = leafID in doc[img]
            inq = leafID in q
            if indoc and inq:
                scores[img] += math.fabs(q[leafID] - doc[img][leafID])
            elif inq and not indoc:
                scores[img] += math.fabs(q[leafID])
            elif not inq and indoc:
                scores[img] += math.fabs(doc[img][leafID])
        if scores[img] <= curr[0]:
            currimg[0], curr[0] = img, scores[img]
    print('done', time.time())
    return currimg[0], curr[0]

def weight(leafID):
    return math.log1p(TOTAL / 1.0 * len(images_in_leaves_table[leafID]))

def match(filename):
    # q is the frequency of this image appearing in each of the leaf nodes
    # count = 0
    q = {}

    try:
        print('start extract', time.time())
        _, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), None)

        print('decorate q', time.time())
        for d in des:
            leafID = lookup(d, 0)
            if leafID in q:
                q[leafID] += 1
            else:
                q[leafID] = 1
        s = 0.0
        for key in q:
            q[key] = q[key] * weight(key)
            s += q[key]
        for key in q:
            q[key] = q[key] / s
        return getScores(q)
    except:
        print("match fail", filename)
    return None, None

with open("model", 'rb') as f:
    node_index, nodes, tree, images_in_leaves_table, avgDepth, doc = pickle.load(f)
    with open("result.csv", "w") as rf:
        count = 0
        for test_file in os.listdir(TEST_DIR):
            count += 1

            if count <= 28:
                continue

            res, score = match(os.path.join(TEST_DIR, test_file))
            if res == None:
                continue

            predict_landmark_id = res.split("@")[0].replace("./train/", "")
            predict_file = res.split("@")[1][:-4]
            
            actual_landmark_id = test_file.split("@")[0]
            actual_file = test_file.split("@")[1][:-4]

            outs = "%s, %s, %s, %s, %s\n" % (predict_landmark_id, predict_file, actual_landmark_id, actual_file,  str(actual_landmark_id == predict_landmark_id))
            rf.write(outs)

            print (count)
            print (outs)

