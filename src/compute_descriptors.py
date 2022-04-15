'''
# Copyright 2021. Dongwon Kim All rights reserved.
#
# File name : compute_descriptors.py
#
# Written by Dongwon Kim
#
# compute descriptors
#   finding image descriptors using visual words
#
# Modificatoin history
#   written by Dongwon Kim on Dec 14, 2021
# 
# notice
#   codeword_center.npy: precomputed centroids of KMean
'''
import numpy as np
import os

SIFT = './sift/sift'
CODEWORD = './codeword_center.npy'
sift_desc = os.listdir(SIFT)
codeword = np.load(CODEWORD)
N = 1000
D = 128

# encode given descriptor
#    for each descriptor in an image,
#        get closest centroid among codeword
#        assign index of codeword as encoded value for each feature
def image_encode(sift_descriptor, codeword):
    encoded = np.zeros(len(sift_descriptor))
    
    for i, feature in enumerate(sift_descriptor):
        distance = np.zeros(D)
        for j, word in enumerate(codeword):
            distance[j] = np.linalg.norm(feature - word)
        index = np.argmin(distance)
        encoded[i] = index
    return encoded

# get weighted descriptors
#    n * log10(D / frequency of the word)
def tf_idf(encoded):
    word_cnt = encoded.sum(axis=0)
    
    weighted = encoded * np.log10(D / word_cnt)
    return weighted


rectype = np.dtype(np.ubyte)
total_encoded = np.zeros((N, D))
for i, feature in enumerate(sift_desc):
    feature = os.path.join(SIFT, feature)
    with open(feature) as f:
        desc = np.fromfile(f, dtype = rectype)
        desc = desc.reshape(-1, 128)
        encoded = image_encode(desc, codeword)
        word, cnt = np.unique(encoded, return_counts=True)

        for j in range(len(word)):
            total_encoded[i][int(word[j])] = cnt[j]
            
    if(i % 100 == 0):
        print("{}/{} encoded".format(i, N))

# compute weighted descriptor using TF IDF
weighted_encoded = tf_idf(total_encoded)

# save descriptors as binary file
variable = np.array([N, D], dtype = np.int32)
weighted_encoded = weighted_encoded.astype(np.float32)
with open('image_descriptor.des', 'wb') as f:
    variable.tofile(f, format="int32")
    weighted_encoded.tofile(f, format="float32")
