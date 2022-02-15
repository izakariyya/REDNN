# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:16:35 2021

@author: 1804499
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve
import memory_profiler
import time

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


# train the classifier

def train_svm(train, trainlabel):
    iotbinclf = GradientBoostingClassifier()
    trainclf = iotbinclf.fit(train, trainlabel)
    return trainclf


def predicted_svm(clf, test):
    predictions = clf.predict(test)
    acc = accuracy_score(testl, predictions)
    return acc


starttb = time.time()
startmb = memory_profiler.memory_usage()

clf = train_svm(train, trainlabel)

endtb = time.time()
endmb = memory_profiler.memory_usage()
train_time_b = endtb - starttb
train_memory_b = endmb[0] - startmb[0]

print("Training time base: {:2f} sec".format(train_time_b))
print("Training memory base: {:2f} mb".format(train_memory_b))


starttesttb = time.time()
starttestmb = memory_profiler.memory_usage()

acc_test_b = predicted_svm(clf, test)

endttestb = time.time()
endtestmb = memory_profiler.memory_usage()
test_time_b = endttestb - starttesttb
test_memory_b = endtestmb[0] - starttestmb[0]

print("Test set accuracy base: {:2f}".format(acc_test_b))
print("Testing time base: {:2f} sec".format(test_time_b))
print("Testing memory base: {:2f} mb".format(test_memory_b))
