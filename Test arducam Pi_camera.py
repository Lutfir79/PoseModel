import cv2
import numpy as np
import time
import PoseModule as pm

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from libcamera import controls

cv2.startWindowThread()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, raw=picam2.sensor_modes[2]))
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

detector = pm.poseDetector()
while True:
    success, img = picam2.capture_array()
    img = cv2.resize(img, (940, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
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
