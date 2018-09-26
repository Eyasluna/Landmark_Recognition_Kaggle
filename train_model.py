from sklearn.cluster import KMeans, MiniBatchKMeans
import os
import pickle
import numpy as np

BRANCHES = 10  # The branching factor in the vocabulary tree
LEAF_CLUSTER_SIZE = 20  # Minimum size of the leaf cluster
MAX_DEPTH = 10
src = './features'

doc = {}
features = []
model = MiniBatchKMeans(n_clusters=BRANCHES)  # The KMeans Clustering Model
count = 0

for file_name in os.listdir(src):
    if count == 100:
        break
    count += 1
    with open(src + '/' + file_name, "rb") as f:
        sub = pickle.load(f)
        print(sub)
        for fea in sub:
            features.append(fea)
    print(file_name)

features = np.array(features)

print(features)

print('done load features')

class Container:
    def __init__(self):
        self.nodes_table = {}
        self.images_in_leaves_table = {}
        self.tree = {}
        self.avg_depth = 0

node_index = 0

def construct_tree(node, container, featuresIDs, depth):
    global node_index
    container.tree[node] = []
    print("tree depth", str(depth))
    if len(featuresIDs) >= LEAF_CLUSTER_SIZE and depth < MAX_DEPTH:
        model.fit([features[i] for i in featuresIDs])

        childFeatureIDs = [[] for i in range(BRANCHES)]
        for i in range(len(featuresIDs)):
            childFeatureIDs[model.labels_[i]].append(featuresIDs[i])
        for i in range(BRANCHES):
            node_index = node_index + 1
            container.nodes_table[node_index] = model.cluster_centers_[i]
            container.tree[node].append(node_index)
            construct_tree(node_index, container, childFeatureIDs[i], depth + 1)
    else:
        container.images_in_leaves_table[node_index] = {}
        container.avg_depth += depth

featuresIDs = [x for x in range(len(features))]
container = Container()

construct_tree(0, container, featuresIDs, 0)

with open("model", "wb") as f:
    pickle.dump((node_index, container.nodes_table, container.tree, container.images_in_leaves_table, container.avg_depth, doc), f)
