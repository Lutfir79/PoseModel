import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (940, 600))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    print(lmList)
    if len(lmList) != 0:
        # RIGHT ARM
        detector.findAngle(img, 12, 14, 16)
        # LEFT ARM
        detector.findAngle(img, 11, 13, 15)

    print(lmList)
    cv2.imshow("Result", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
