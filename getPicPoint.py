# coding=utf-8
import os
import sys
import cv2
import numpy
import dlib
import json

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def read_im(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1],
                         im.shape[0]))
    return im


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise Exception
    if len(rects) == 0:
        raise Exception

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    points = []
    for idx, point in enumerate(landmarks):
        points.append((point[0, 0], point[0, 1]))
    return points


d = "FaceEmotion\\img\\"
emo = "FaceEmotion\\emo\\"


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            allfiles.append(filepath)
    return allfiles


l = GetFileFromThisRootDir(d)
itemFile = open('item.points', 'w')
for i in l:
    i = str(i)
    emoPath = i[0:i.rfind('.')].replace('img', 'emo') + ".emo"
    print emoPath
    emoData = open(emoPath, 'r').read()
    emoVals = [float(x) for x in emoData.split(' ')]
    label = emoVals.index(max(emoVals))

    im = read_im(i)
    try:
        pts = annotate_landmarks(im, get_landmarks(im))
        print pts
        itemFile.write(str(label) + json.dumps(pts) + "\n")
    except:
        pass
