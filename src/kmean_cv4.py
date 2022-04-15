'''
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : kmean_cv4.py
#
# Written by Dongwon Kim
#
# image descriptor
#   find global visual words using centroids of KMean 
#
# Modificatoin history
#   written by Dongwon Kim on Dec 14, 2021
#
# using google colab to compute KMean
# this code works with google.colab
# 
'''

from google.colab import drive, files
import numpy as np
import os
from sklearn.cluster import KMeans

drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/Colab Notebooks/sift'

os.chdir(PATH)

sift_feature = os.listdir(PATH)
rectype = np.dtype(np.ubyte)

f = open(sift_feature[0], 'rb')
desc = np.fromfile(f, dtype=rectype)

desc = desc.reshape(-1, 128)
print(desc)

f.close()
descriptors = desc

# stack feature from all images as one matrix
for features in sift_feature[1:]:
    with open(features) as f:
        desc = np.fromfile(f, dtype=rectype)
        desc = desc.reshape(-1, 128)
        descriptors = np.vstack((descriptors, desc))

descriptors = np.array(descriptors)

k = 128
km = KMeans(n_clusters = k)
km.fit(descriptors)

voc_center = km.cluster_centers_

np.save('../codeword_center', voc_center)
