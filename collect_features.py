# Author: Yibo Fu
# 6/2/2018

import os
import pickle
import numpy as np

features = []

for file in os.listdir('./features'):
    fin = open('./features/' + file, 'rb')
    des = pickle.load(fin)
    for d in des:
        features.append(d)
    
np_features = np.array(features)

with open('all_features', 'wb') as f:
    pickle.dump(np_features, f)



