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

        # K-Means for code summarization
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features)
        # kmeans = KMeans(n_clusters=10, random_state=0).fit(scaled_features)
        # kmeans_features = np.column_stack([np.sum((features - center) ** 2, axis=1) ** 0.5 for center in kmeans.cluster_centers_])
        # features_mat[:, i] = np.mean(kmeans_features, axis=1)

        # pca for dimension reduction
        # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=0.95))])
        # pca = pipeline.fit_transform(features)
        # features_mat[:, i] = np.mean(pca, axis=1)

        # mean pooling
        features_mat[:, i] = np.mean(features, axis=1)

        i = i + 1

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








