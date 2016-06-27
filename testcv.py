# coding=utf-8
from svmutil import *
import cv2
import numpy as np
from flask import Flask, render_template
from flask_sockets import Sockets
import dlib

app = Flask(__name__)
sockets = Sockets(app)
classfier = cv2.CascadeClassifier('/home/nlfox/soft/opencv-2.4.13/data/haarcascades/haarcascade_frontalface_alt.xml')
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

m_sex = svm_load_model("sex.model")
m_emo = svm_load_model("libsvm.model")

map_emo = {0: "anger",
           1: "contempt",
           2: "disgust",
           3: "fear",
           4: "happiness",
           5: "neutral",
           6: "sadness",
           7: "surprise"}


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise Exception
    if len(rects) == 0:
        raise Exception

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    points = []
    for idx, point in enumerate(landmarks):
        points.append((point[0, 0], point[0, 1]))
    return points


@sockets.route('/echo')
def echo_socket(ws):
    while not ws.closed:
        message = ws.receive()
        print message, ws.closed
        tmp = message.split(',')[1]
        im = str(tmp.decode("base64"))
        nparr = np.fromstring(im, np.uint8)
        img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
        size = img.shape[:2]
        divisor = 8
        h, w = size
        minSize = (w / divisor, h / divisor)
        rects = classfier.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)
        try:
            x, y, w, h = rects[0]
        except:
            ws.send("no face")
            continue

        pts = annotate_landmarks(img, get_landmarks(img))
        print pts
        pointsComb = []
        firstPoint = pts[0]
        pointsComb.append(0)
        pointsComb.append(0)
        for point in pts[1:]:
            pointsComb.append(point[0] - firstPoint[0])
            pointsComb.append(point[1] - firstPoint[1])
        lineMax = float(max(pointsComb))
        xi = {}
        ind = 0
        for e in pointsComb:
            ind += 1
            val = e
            xi[ind] = val / lineMax
        prob_x = [xi]
        emo = svm_predict([1] * len(prob_x), prob_x, m_emo)[0]

        # --------
        img = img[y:y + h, x:x + w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (40, 40))

        lineArr = [img[i, j] for i in xrange(40) for j in xrange(40)]
        # hog = cv2.HOGDescriptor((40, 40), (16, 16), (8, 8), (8, 8), 9)
        # h = hog.compute(img)
        # res = [i[0] for i in h]
        features = lineArr[0:-1]
        xi = {}
        ind = 0
        for e in features:
            ind += 1
            val = e
            xi[ind] = float(val) / 255
        prob_x = [xi]
        sex = svm_predict([1] * len(prob_x), prob_x, m_sex)[0]
        res = "女" if sex == [0.0] else "男"
        # im = numpy.array(message.split(",")[1].decode("base64"))
        # img=cv2.imdecode(im, 1)
        # cv2.imshow("img",img)
        res += str(map_emo[int(emo[0])])
        ws.send(res)


@app.route('/')
def hello():
    return render_template("index.html")


if __name__ == "__main__":
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
