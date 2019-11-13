import cv2
import os

def camera_capture():
   cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink" , cv2.CAP_GSTREAMER)

   dim=(32,32)

   if cap.isOpened():
     ret,frame=cap.read()
     cv2.imwrite('image.jpg',frame)
     roi=frame[15:520,265:800]
     roi=cv2.resize(roi,dim,cv2.INTER_AREA)
     roi=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
     for i in range(3):
        roi[:,:,i] = cv2.equalizeHist(roi[:,:,i]) # histogram equalisation
     cv2.imwrite('roi.jpg',roi)
     prediction = os.popen('curl -X POST 0.0.0.0:81/predict -F "imagefile=@roi.jpg"').read()
     print("Predcited label: ", prediction)
   else:
      print('camera failed to open')
   cap.release()

