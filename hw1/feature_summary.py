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

data_path = './dataset/'
features_path = './features/'
DIM = 293

def mean_features(dataset='train'):
    
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

        # mean pooling
        # temp = np.mean(features, axis=1)
        # features_mat[:,i] = np.mean(features, axis=1)
        # i = i + 1

        # kmeans for codebook summarization
        kmeans = KMeans(n_clusters=5, random_state=0).fit(features)
        for i in range(features.shape[0]):
            features_mat[i,:] = kmeans.cluster_centers_[kmeans.labels_[i]]

        print(kmeans)

    f.close()

    return features_mat
        
if __name__ == '__main__':
    train_data = mean_features('train')
    valid_data = mean_features('valid')

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.savefig("feature_summary.png")








