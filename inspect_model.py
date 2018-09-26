import pickle

with open("model", 'rb') as f:
    node_index, nodes, tree, images_in_leaves_table, avgDepth, doc = pickle.load(f)
    print(len(images_in_leaves_table.keys()))