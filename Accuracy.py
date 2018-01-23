# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:43:18 2017

@author: dev1
"""

import pickle
import sys
import matplotlib.pyplot as plt

test_file = r"C:\work\py\cat\Accuracy_test.dump"
svm_file = r"C:\work\py\cat\svm_output.dump"
conf_output_file = r"C:\work\py\cat\Accuracy_test_conf.dump"

classifier = pickle.load(open(svm_file, "rb"))
X, y = pickle.load(open(test_file, "rb"))

y_predict = classifier.predict(X)
correct = 0
for i in range(len(y)):
    if y[i] == y_predict[i]:
        correct += 1
print('Accuracy: %f' % (float(correct) / len(y)))


X2, y2 = pickle.load(open(conf_output_file, "rb"))
for i in range(len(y)):
    if y[i] != y_predict[i]:
        print("miss data is " + str(X2[i]))


