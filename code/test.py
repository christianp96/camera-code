import os
import cv2
import numpy as np
import subprocess

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink" , cv2.CAP_GSTREAMER)

dim=(32,32)

if cap.isOpened():
  while(True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
      cv2.imwrite('image.jpg',frame)
      roi=frame[15:520,265:770]
      roi=cv2.resize(roi,dim,cv2.INTER_AREA)
      for i in range(3):
        roi[:,:,i] = cv2.equalizeHist(roi[:,:,i]) # histogram equalisation
      cv2.imwrite('roi.jpg',roi)
      prediction = os.popen('curl -X POST 0.0.0.0:81/predict -F "imagefile=@roi.jpg"').read()
      print("Predcited label: ", prediction)
      break
else:
  print('camera failed to open')
cap.release()
cv2.destroyAllWindows()
