# GCT634 (2018) HW1
# Mar-18-2018: initial version
# Juhan Nam
#
# Sept-20-2022: edited version
# Vanessa Tan

import sys
import os
import numpy as np
import librosa
from feature_summary import *
import random

from sklearn import model_selection
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_model(train_X, train_Y, valid_X, valid_Y, model, name):

    # train
    if (name == 'Kmeans') or (name == 'GMM'):
        model.fit(train_X)
    else:
        model.fit(train_X, train_Y)

    # validation
    valid_Y_hat = model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print(name + ': validation accuracy = ' + str(accuracy) + ' %')

    return model, accuracy

if __name__ == '__main__':

    random.seed(0)

    # load data 
    train_X = mean_features('train')
    valid_X = mean_features('valid')

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # prepare models
    models = []
    models.append(('SGD', SGDClassifier(random_state=0)))
    models.append(('SVM', SVC(random_state=0)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('RF', RandomForestClassifier(random_state=0)))

    # evaluate each model
    valid_acc = []
    names = []
    model_final = []
    for name, model in models:
        model, accuracy = train_model(train_X, train_Y, valid_X, valid_Y, model, name)
        valid_acc.append(accuracy)
        names.append(name)
        model_final.append(model)

    # choose the model that achieve the best validation accuracy
    final_model = model_final[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    print('best classifier = ' + str(names[np.argmax(valid_acc)]))
    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('final validation accuracy = ' + str(accuracy) + ' %')



