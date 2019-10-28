from src.load_model import LoadedModel
from src.preprocess import preprocess_greyscale
import cv2
import numpy as np

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

sessionSavePath = "./session"
model=LoadedModel(sessionSavePath)
dim=(32,32)

if cap.isOpened():
  while(True):
    ret,frame=cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
      cv2.imwrite('image.jpg',frame)
      roi=frame[15:520,265:745]
      cv2.imwrite('roi.jpg',roi)
      c=cv2.resize(c,dim,cv2.INTER_AREA)
      #cv2.imwrite('image.jpg',c)
      c=[cv2.cvtColor(c,cv2.COLOR_BGR2RGB)]
      c=np.asarray(c,dtype=np.uint8)
      prediction = model.predict(preprocess_greyscale(c))
      print(np.argmax(prediction))
      break
else:
  print('camera failed to open')
cap.release()
cv2.destroyAllWindows()
