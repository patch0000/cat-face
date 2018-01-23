# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 09:00:46 2017

@author: dev1
"""

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
                    # lbpは画像の特徴量がピクセル単位でそのまま入った配列（64 x 64）
                    # 各セル(8x8)にそれぞれの特徴量（0:LBP_POINTS+2）が何回現れたかを
                    # カウントし、histogramに格納する
                    histogram[int(lbp[y + dy, x + dx])] += 1
            histograms.append(histogram)
    # 一次元配列を返す 26 x 8 x 8 = 1664
    return np.concatenate(histograms)


def get_features(directory, rFlag):
    features = []
    f_dict = {}
    for fn in iglob('%s/*.png' % directory):
        image = color.rgb2gray(io.imread(fn))
        lbp_image = feature.local_binary_pattern(
            image, LBP_POINTS, LBP_RADIUS, 'uniform')
        features.append(get_histogram_feature(lbp_image))
        f_dict[fn] = get_histogram_feature(lbp_image)

        # サンプル水増しのために反転イメージを追加
        if rFlag is True:
            lbp_image = feature.local_binary_pattern(
                    np.fliplr(image), LBP_POINTS, LBP_RADIUS, 'uniform')
            features.append(get_histogram_feature(lbp_image))
        """
        f_dict['flip_' + fn] = get_histogram_feature(lbp_image)
        """
    return features, f_dict


def main():

    # 本番時　(19310, 1664) (19310,)
    positive_dir = r'C:\work\py\cat\positive'
    negative_dir = r'C:\work\py\cat\negative'
    output_file = r'C:\work\py\cat\image_feature_for_svm.dump'

    positive_samples, p_dict = get_features(positive_dir, True)
    negative_samples, n_dict = get_features(negative_dir, True)
    n_positives = len(positive_samples)
    n_negatives = len(negative_samples)
    X = np.array(positive_samples + negative_samples)
    y = np.array([1 for i in range(n_positives)] +
                 [0 for i in range(n_negatives)])
    print(X.shape, y.shape)
    pickle.dump((X, y), open(output_file, 'wb'))

    # 精度測定サンプル作成時(211, 1664) (211,)
    test_positive_dir = r"C:\work\py\cat\positive_test"
    test_negative_dir = r"C:\work\py\cat\negative_test"
    test_output_file = r"C:\work\py\cat\Accuracy_test.dump"
    conf_output_file = r"C:\work\py\cat\Accuracy_test_conf.dump"

    test_positive_samples, test_p_dict = get_features(test_positive_dir, False)
    test_negative_samples, test_n_dict = get_features(test_negative_dir, False)

    test_n_positives = len(test_positive_samples)
    test_n_negatives = len(test_negative_samples)
    X = np.array(test_positive_samples + test_negative_samples)
    y = np.array([1 for i in range(test_n_positives)] +
                 [0 for i in range(test_n_negatives)])
    print(X.shape, y.shape)
    pickle.dump((X, y), open(test_output_file, 'wb'))

    # どのファイルが間違ったかわかるように精度測定サンプルのファイル名を保存しておく
    # (211,) (211, 1664)
    w = list()
    z = list()
    for i in test_p_dict.keys():
        z.append(i)
        w.append(test_p_dict[i])

    for i in test_n_dict.keys():
        z.append(i)
        w.append(test_n_dict[i])

    X = np.array(z)
    y = np.array(w)
    print(X.shape, y.shape)
    pickle.dump((X, y), open(conf_output_file, 'wb'))

if __name__ == "__main__":
    main()
