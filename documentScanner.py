import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
kernel=np.ones((5,5))
cam = cv2.VideoCapture(0)
cam.set(3, frameWidth)
cam.set(4, frameHeight)

def preprocessing(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(7,7),2)
    imgCanny=cv2.Canny(imgBlur,100,100)
    imgDilate=cv2.dilate(imgCanny,kernel,iterations=2)
    imgErode=cv2.erode(imgDilate,kernel,iterations=2)

    return imgErode


while True:
    success, img = cam.read()
    img=cv2.resize(img,(frameWidth,frameHeight))
    imgProcess=preprocessing(img)
    cv2.imshow('Scanner', imgProcess)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
