# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:17:49 2018

@author: dev1
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pickle
import sys
import pandas as pd
import seaborn as sns


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
#        plt.scatter(range(len(X[0])), X[0], color='red')
print('Accuracy: %f' % (float(correct) / len(y)))


X2, y2 = pickle.load(open(conf_output_file, "rb"))
for i in range(len(y)):
    if y[i] != y_predict[i]:
        print("miss data is " + str(X2[i]))
#        plt.scatter(range(len(X[i])), X[i],  color='blue')


# グラフを表示する
"""
plt.autoscale()
plt.grid()
plt.show()
""" 