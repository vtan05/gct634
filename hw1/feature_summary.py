# GCT634 (2018) HW1
# Mar-18-2018: initial version
# Juhan Nam
#
# Sept-20-2022: edited version
# Vanessa Tan

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from time import time

data_path = './dataset/'
features_path = './features/'
# DIM = 293 * 2 # for PCA and KMeans
DIM = 293 # for Mean

def pool_features(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        features_mat = np.zeros(shape=(DIM, 1100))
    else:
        features_mat = np.zeros(shape=(DIM, 300))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        features_file = features_path + file_name
        features = np.load(features_file)
        # print(features.shape)

        # mean pooling
        features_mat[:, i] = np.mean(features, axis=1)

        # other pooling
        # features_mat[:, i] = features.flatten()

        i = i + 1

    f.close()

    return features_mat
        
if __name__ == '__main__':
    train_data = pool_features('train')
    valid_data = pool_features('valid')

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.savefig("feature_summary.png")








