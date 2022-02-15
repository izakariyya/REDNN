# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:16:35 2021

@author: 1804499
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
import random


#Function to load, normalize and splits datasets in csv format

def data():
    benign = np.loadtxt("benign_traffic.csv", delimiter = ",")
    mirai = np.loadtxt("mirai_traffic.csv", delimiter = ",")
    gafgyt = np.loadtxt("gafgyt_traffic.csv", delimiter = ",")
    alldata = np.concatenate((benign, gafgyt, mirai))
    j = len(benign[0])
    data = alldata[:, 1:j] 
    benlabel = alldata[:, 0]
    bendata = (data - data.min()) / (data.max() - data.min())
    bendata, benmir, benlabel, benslabel = train_test_split(bendata, benlabel, test_size = 0.2, random_state = 42)
    return bendata, benmir, benlabel, benslabel

#load data

train, test, trainlabel, testl = data()

trainlbl = trainlabel.tolist()

n = len(trainlbl)
percent = 0.5

partly_flipped = [1.0 - trainlbl[i] if i in random.sample(list(range(n)),int(percent*n)) else trainlbl[i] for i in range(n) ]

partly_flipped = np.asarray(partly_flipped)


# train the classifier

def train_gbdt(train, trainlabel):
    iotbinclf = GradientBoostingClassifier()
    trainclf = iotbinclf.fit(train, trainlabel)
    return trainclf


def predicted_gbdt(clf, test):
    predictions = clf.predict(test)
    acc = accuracy_score(testl, predictions)
    return acc


clf = train_gbdt(train, partly_flipped)

acc_test_b = predicted_gbdt(clf, test)

print("Test set accuracy base: {:2f}".format(acc_test_b))