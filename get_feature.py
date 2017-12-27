# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:00:46 2017

@author: dev1
"""

import sys
import numpy as np
from skimage import io, feature, color
from glob import iglob
import pickle

WIDTH, HEIGHT = (64, 64)
LBP_POINTS = 24
LBP_RADIUS = 3
cell_size = 8


def get_histogram_feature(lbp):

    bins = LBP_POINTS + 2
    histograms = []
    for y in range(0, HEIGHT, cell_size):
        for x in range(0, WIDTH, cell_size):
            histogram = np.zeros(shape=(bins,))
            for dy in range(cell_size):
                for dx in range(cell_size):
                    #  print(x, dx, y, dy)

                    # lbpは画像の特徴量がそのまま入った配列（64 x 64）
                    # histogramは各セルの特徴量合計（26要素の配列、LBP_POINTS+2）
                    histogram[int(lbp[y + dy, x + dx])] += 1
            histograms.append(histogram)
    # 64x64画像の特徴を一次元配列にして返す 24 x 8 x 8 = 1664
    return np.concatenate(histograms)


def get_features(directory):
    features = []
    for fn in iglob('%s/*.png' % directory):
        image = color.rgb2gray(io.imread(fn))
        lbp_image = feature.local_binary_pattern(
            image, LBP_POINTS, LBP_RADIUS, 'uniform')
        features.append(get_histogram_feature(lbp_image))

        # サンプル水増しのために反転イメージを追加
        lbp_image = feature.local_binary_pattern(
                np.fliplr(image), LBP_POINTS, LBP_RADIUS, 'uniform')
        features.append(get_histogram_feature(lbp_image))
    return features


def main():

    # 精度測定サンプル作成時 (2240, 1664) (2240,)
    positive_dir = r"C:\work\py\cat\positive_test"
    negative_dir = r"C:\work\py\cat\negative_test"
    output_file = r"C:\work\py\cat\Accuracy_test.dump"

    # 本番時　(19732, 1664) (19732,)
    # positive_dir = r'C:\work\py\cat\positive'
    # negative_dir = r'C:\work\py\cat\negative'
    # output_file = r'C:\work\py\cat\image_feature_for_svm.dump'

    positive_samples = get_features(positive_dir)
    negative_samples = get_features(negative_dir)
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)
    X = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_positives)] +
                 [0 for i in range(n_negatives)])
    print(X.shape, y.shape)
    pickle.dump((X, y), open(output_file, 'wb'))

if __name__ == "__main__":
    main()
