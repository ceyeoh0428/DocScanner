import cv2
import numpy as np

widthImg = 540
heightImg = 640
kernel = np.ones((5, 5))

cam = cv2.VideoCapture(0)


# cam.set(3, widthImg)
# cam.set(4, heightImg)


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel, iterations=1)

    return imgErode


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for shapes in contours:
        area = cv2.contourArea(shapes)
        if area > 2500:
            # cv2.drawContours(imgContour, shapes, -1, (0, 0, 255), 3)
            perimeter = cv2.arcLength(shapes, True)
            approxCorner = cv2.approxPolyDP(shapes, 0.02 * perimeter, True)
            if area > maxArea and len(approxCorner) == 4:
                biggest = approxCorner
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 20)
    return biggest


def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    # print('add',add)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    # print('newpoints',newpoints)

    return newPoints


def wrap(img, biggest):
    biggest = reorder(biggest)
    p1 = np.float32(biggest)
    p2 = np.float32([0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg])
    matrix = cv2.getPerspectiveTransform(p1, p2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


while True:
    success, img = cam.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgProcess = preprocessing(img)
    biggest = getContours(imgProcess)
    # wrap(img,biggest)
    print(biggest)
    cv2.imshow('Scanner', imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
