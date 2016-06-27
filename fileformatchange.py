import cv2
import numpy as np
import json
prob_y = []
prob_x = []
data_file_name = 'sex_pic.txt'
converted = open("out.txt", "w+")
for line in open(data_file_name):
    lineArr = line.split(' ')
    label = lineArr[-1]
    features = lineArr[0:-1]
    hog = cv2.HOGDescriptor((40, 40), (16, 16), (8, 8), (8, 8), 9)
    img = np.zeros((40, 40, 1), np.uint8)

    for i in xrange(40):
        for j in xrange(40):
            img[i, j] = lineArr[i * 40 + j]

    h = hog.compute(img)

    xi = {}
    ind = 0
    for e in h:
        ind += 1
        val = e[0]
        xi[ind] = float(val)
    prob_y += [float(label)]
    prob_x += [xi]
    converted.write(label.strip() + json.dumps(xi) + "\n")
