import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
dispW = 720
dispH = 480
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

fps = 0
pos = (30, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
height = 1.5
weight = 3
myColor = (0, 0, 255)

faceCascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

while True:
    tStart = time.time()
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frameGray, 1.3, 5)

    # Print "Face detected" only if there are faces
    if len(faces) > 0:
        print("Face detected")
    else:
        print("Face not detected")

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roiGray = frameGray[y:y + h, x:x + w]
        roiColor = frame[y:y + h, x:x + w]

    cv2.putText(frame, str(int(fps)) + ' FPS', pos, font, height, myColor, weight)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = 0.9 * fps + 0.1 * (1 / loopTime)

cv2.destroyAllWindows()

