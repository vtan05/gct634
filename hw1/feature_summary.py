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

        # check optimal clusters for unsupervised learning models
        # silhouette = []
        # for n_clusters in range(2,11):
        #     clusters = KMeans(n_clusters=n_clusters, random_state=0)
        #     labels = clusters.fit_predict(features)
        #     silhouette_avg = silhouette_score(features, labels)
        #     print("n_clusters =", n_clusters,
        #         "silhouette_score :",silhouette_avg)

        # K-Means for code summarization
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features)
        # kmeans = KMeans(n_clusters=10, random_state=0).fit(scaled_features)
        # kmeans_features = np.zeros(scaled_features.shape)
        # for i in range(features.shape[0]):
        #      kmeans_features[i,:] = kmeans.cluster_centers_[kmeans.labels_[i]]

        # pca for dimension reduction
        # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=0.95))])
        # pca = pipeline.fit_transform(features)

        # mean pooling
        features_mat[:, i] = np.mean(kmeans_features, axis=1)

        # max pooling
        # features_mat[:,i] = np.max(features, axis=1)

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








