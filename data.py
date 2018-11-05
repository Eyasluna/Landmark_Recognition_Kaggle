# coding: utf-8
# Author: Yibo Fu
# 6/2/2018
import os
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle

featuresIDs  = []
features = []
TOTAL = 7500
TRAIN_FILE = "train.csv"
nodes = {}  # List of nodes (list of SIFT descriptors)
nodeIndex = 0  # Index of the last node for which subtree was constructed
tree = {}  # A dictionary in the format - node: [child1, child2, ..]
branches = 10  # The branching factor in the vocabulary tree
leafClusterSize = 20  # Minimum size of the leaf cluster
imagesInLeaves = {}  # Dictionary in the format - leafID: [img1:freq, img2:freq, ..]
doc = {}
maxDepth = 10
avgDepth = 0
model = MiniBatchKMeans(n_clusters=branches)  # The KMeans Clustering Model
sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)  # SIFT Feature extractor model


def read_all_landmark_id():
    all_landmark_id = set()
    with open(TRAIN_FILE) as f:
        f.readline()
        for line in f:
            _, url, landmark_id = line.replace("\n", "").split(",")
            all_landmark_id.add(landmark_id)
    return all_landmark_id


def read_all_landmark_pictures():
    landmark_id_files = {}
    for f in os.listdir("./train"):
        if f.endswith(".jpg"):
            landmark_id = f.split("@")[0]
            if landmark_id not in landmark_id_files:
                landmark_id_files[landmark_id] = []
            landmark_id_files[landmark_id].append(f)
    return landmark_id_files


def dumpFeature():
    features = []
    count = 0
    for fname in os.listdir("./train"):
        try:
            kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread("./train" + "/" + fname),
                                                         cv2.COLOR_BGR2GRAY), None)
            for d in des:
                features.append(d)
        except Exception as e:
            pass
        count += 1
        if count % 100 == 0:
            print(count)
    features = np.array(features)
    return features


def constructTree(node, featuresIDs, depth):
    global nodeIndex, nodes, tree, imagesInLeaves, avgDepth
    tree[node] = []
    if len(featuresIDs) >= leafClusterSize and depth < maxDepth:
        model.fit([features[i] for i in featuresIDs])
        childFeatureIDs = [[] for i in range(branches)]
        for i in range(len(featuresIDs)):
            childFeatureIDs[model.labels_[i]].append(featuresIDs[i])
        for i in range(branches):
            nodeIndex = nodeIndex + 1
            nodes[nodeIndex] = model.cluster_centers_[i]
            tree[node].append(nodeIndex)
            constructTree(nodeIndex, childFeatureIDs[i], depth + 1)
    else:
        imagesInLeaves[node] = {}
        avgDepth = avgDepth + depth


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


def tfidf(filename):
    global imagesInLeaves
    kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), None)
    for d in des:
        leafID = lookup(d, 0)
        if filename in imagesInLeaves[leafID]:
            imagesInLeaves[leafID][filename] += 1
        else:
            imagesInLeaves[leafID][filename] = 1


def dump_model():
    global nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc
    with open("model", "wb") as f:
        pickle.dump((nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc), f)


def load_model():
    with open("model", "rb") as f:
        nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc = pickle.load(f)
        return nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc


def weight(leafID):
    return math.log1p(TOTAL / 1.0 * len(imagesInLeaves[leafID]))


def do_tf_idf():
    for fname in os.listdir("./train"):
        try:
            tfidf('./train/' + fname)
        except Exception as e:
            pass

    for leafID in imagesInLeaves:
        for img in imagesInLeaves[leafID]:
            if img not in doc:
                doc[img] = {}
            doc[img][leafID] = weight(leafID) * (imagesInLeaves[leafID][img])

    for img in doc:
        s = 0.0
        for leafID in doc[img]:
            s += doc[img][leafID]
        for leafID in doc[img]:
            doc[img][leafID] /= s


def getScores(q):
    scores = {}
    curr = [float("inf")]
    currimg = [""]
    for fname in os.listdir("./train/"):
        img = "./train/"+fname
        scores[img] = 0
        if img not in doc:
            continue
        for leafID in imagesInLeaves:
            if leafID in doc[img] and leafID in q:
                scores[img] += math.fabs(q[leafID] - doc[img][leafID])
            elif leafID in q and leafID not in doc[img]:
                scores[img] += math.fabs(q[leafID])
            elif leafID not in q and leafID in doc[img]:
                scores[img] += math.fabs(doc[img][leafID])
            # if scores[img] > curr[-1]:
            #     break
        if scores[img] <= curr[0]:
            currimg[0], curr[0] = img, scores[img]

    return currimg[0], curr[0]


def match(filename):
    # q is the frequency of this image appearing in each of the leaf nodes
    count = 0
    q = {}
    kp, des = sift.detectAndCompute(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), None)
    for d in des:
        count += 1
        if count % 100 == 0:
            print(count)
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


def load_all():
    global nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc, features, featuresIDs
    with open("all_features", "rb") as f:
        features = pickle.load(f)
    print(len(features))
    root = features.mean(axis=0)
    nodes[0] = root
    featuresIDs = [x for x in range(len(features))]
    nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc = load_model()

if __name__ == '__main__':
    all_ids = read_all_landmark_id()
    print(len(all_ids))
    pics = read_all_landmark_pictures()
    print(len(pics))
    features = dumpFeature()
    with open("all_features", "wb") as f:
        pickle.dump(features, f)
    with open("all_features", "rb") as f:
        features = pickle.load(f)
    print(len(features))
    root = features.mean(axis=0)
    nodes[0] = root
    featuresIDs = [x for x in range(len(features))]
    constructTree(0, featuresIDs, 0)
    dump_model()
    print("end construct tree")
    #nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc = load_model()
    do_tf_idf()
    print("end tf-idf")
    dump_model()
    # nodeIndex, nodes, tree, imagesInLeaves, avgDepth, doc = load_model()
    # x = match("./train/14390@8a5f41fb6efa27b8.jpg")
    # print(x)
