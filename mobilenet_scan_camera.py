import argparse
import cv2
from cv2 import dnn
import numpy as np

#inWidth = 300
#inHeight = 300
inWidth = 224
inHeight = 224
WHRatio = inWidth / float(inHeight)
#inScaleFactor = 0.007843
inScaleFactor = 0.017
meanVal = (103.94, 116.78, 123.68)
#meanVal = (127.5, 127.5, 127.5)
#meanVal = 127.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="number of video device", default=0)
    parser.add_argument("--prototxt", default="mobilenet_v2_deploy.prototxt")
    parser.add_argument("--caffemodel", default="mobilenet_v2.caffemodel")
    parser.add_argument("--classNames", default="synset.txt")
    #parser.add_argument("--thr", default=0.2, help="confidence threshold to filter out weak detections")
    args = parser.parse_args()
    net = dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
    cap = cv2.VideoCapture(args.video)
    f = open(args.classNames, 'r')
    classNames = f.readlines()

    while True:
        ret, frame = cap.read()
        print(frame.shape)
        WHRatio = inWidth / float(inHeight)

        #grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #blob = dnn.blobFromImage(grayFrame, inScaleFactor, (inWidth, inHeight), (104, 117, 123))
        blob = dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal)
        net.setInput(blob)
        detections = net.forward()

        maxClassId = 0
        maxClassPoint = 0;
        # reshape and get max
        #detectMat = detections.reshape(1, 1)
        #print(detectMat.shape)
        for i in range(detections.shape[1]):
            classPoint = detections[0, i, 0, 0]
            if (classPoint > maxClassPoint):
                maxClassId = i
                maxClassPoint = classPoint

        print("class id: ", maxClassId)
        print("class point: ", maxClassPoint)
        print("name: ", classNames[maxClassId])

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) >= 0:
            break
