import cv2
import time
import PoseModule as pm
cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList[1])
    cv2.circle(img, (lmList[1][1], lmList[1][2]), 15, (0, 0, 255), cv2.FILLED)
    # PUT FPS TO
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (78, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Imager", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()