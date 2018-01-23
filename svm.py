# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:34:26 2017

@author: dev1
"""

import sys
import sklearn.svm
import pickle

input_file = r"C:\work\py\cat\image_feature_for_svm.dump"
output_file = r"C:\work\py\cat\svm_output.dump"

X, y = pickle.load(open(input_file, "rb"))
classifier = sklearn.svm.LinearSVC(C=0.0001)
classifier.fit(X, y)
pickle.dump(classifier, open(output_file, "wb"))
