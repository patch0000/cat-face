# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:12:40 2017

@author: dev1
"""
import pickle
import numpy as np
from skimage import io, color, transform, feature
import matplotlib.pyplot as plt
import matplotlib.cm as cm

WIDTH, HEIGHT = (64, 64)
CELL_SIZE = 8
THRESHOLD = 3.0
LBP_POINTS = 24
LBP_RADIUS = 3

input_file = r'C:\work\py\cat\svm_output.dump'
target_file = r'\work\py\cat\ng.png'
# target_file = r'\work\py\cat\ok.png'


def overlap_score(a, b):
    left = max(a['x'], b['x'])
    right = min(a['x'] + a['width'], b['x'] + b['width'])
    top = max(a['y'], b['y'])
    bottom = min(a['y'] + a['height'], b['y'] + b['height'])
    intersect = max(0, (right - left) * (bottom - top))
    union = a['width'] * a['height'] + b['width'] * b['height'] - intersect
    return intersect / union


def get_histogram(image):
    hCELL_SIZE = CELL_SIZE
    lbp = feature.local_binary_pattern(image,
                                       LBP_POINTS, LBP_RADIUS, 'uniform')
    bins = LBP_POINTS + 2
    histogram = np.zeros(shape=(int(image.shape[0] / hCELL_SIZE),
                                int(image.shape[1] / hCELL_SIZE), bins),
                         dtype=np.int)

    for y in range(0, image.shape[0] - hCELL_SIZE, hCELL_SIZE):
        for x in range(0, image.shape[1] - hCELL_SIZE, hCELL_SIZE):
            for dy in range(hCELL_SIZE):
                for dx in range(hCELL_SIZE):
                    # print(x, dx, y, dy)
                    histogram[int(y / hCELL_SIZE),
                              int(x / hCELL_SIZE),
                              int(lbp[y + dy, x + dx])] += 1

    return histogram


svm = pickle.load(open(input_file, "rb"))
target = color.rgb2gray(io.imread(target_file))
target_scaled = target + 0

scale_factor = 2.0 ** (-1.0 / 8.0)
detections = []

for s in range(16):
    histogram = get_histogram(target_scaled)

    for y in range(0, int(histogram.shape[0] - HEIGHT / CELL_SIZE)):
        for x in range(0, int(histogram.shape[1] - WIDTH / CELL_SIZE)):
            myfeature = histogram[y:y + int(HEIGHT / CELL_SIZE),
                                  x:x + int(WIDTH / CELL_SIZE)].reshape(1, -1)
            score = svm.decision_function(myfeature)
            if score[0] > THRESHOLD:
                scale = (scale_factor ** s)
                detections.append({
                        'x': x * CELL_SIZE / scale,
                        'y': y * CELL_SIZE / scale,
                        'width': WIDTH / scale,
                        'height': HEIGHT / scale,
                        'score': score
                        })
    target_scaled = transform.rescale(target_scaled, scale_factor,
                                      mode='constant')


detections = sorted(detections, key=lambda d: d['score'], reverse=True)
deleted = set()
for i in range(len(detections)):
    if i in deleted:
        continue
    for j in range(i + 1, len(detections)):
        if overlap_score(detections[i], detections[j]) > 0.3:
            deleted.add(j)

detections = [d for i, d in enumerate(detections) if i not in deleted]

fig, (ax1) = plt.subplots(ncols=1)  # ,figsize=(10, 5))
for i in range(1):
    j = i
    ax1.add_patch(plt.Rectangle((detections[j]['y'],
                                 detections[j]['x']),
                                detections[j]['width'],
                                detections[j]['height'],
                                edgecolor='w',
                                facecolor='none',
                                linewidth=2.5))

ax1.imshow(target, cmap=cm.Greys_r)
