import os

import cv2
classfier = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')

def read_im(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    size=im.shape[:2]
    divisor=8
    h, w = size
    minSize=(w/divisor, h/divisor)
    rects =  classfier.detectMultiScale(im, 1.2, 2, cv2.CASCADE_SCALE_IMAGE,minSize)
    n = 0
    for i in rects:
        x, y, w, h = i
        new_im = im[y:y+h, x:x+w]

        cv2.imwrite("split\\"+fname[:-4] + "_" + str(n) +".jpg", new_im)
        n+=1
d = "training-synthetic"

def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


l = GetFileFromThisRootDir(d, ext='pgm')
for i in l:
    print i
    im = read_im(i)