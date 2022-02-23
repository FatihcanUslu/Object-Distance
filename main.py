
import cv2
import cvzone
import numpy as np
from numba import jit, cuda
import time

cap=cv2.VideoCapture("videolar/testvideo.mp4")


# used to record the time when we processed last frame(for calculating fps)
prev_frame_time = 0
# used to record the time at which we processed current frame(for calculating fps)
new_frame_time = 0

def drawDistance(radius,x,y):
    w=radius#small w = weight in pixels
    W = 6.3#big W =weight in real life
    f = 1750
    d = (W * f) / w  # calculated distance
    #f=(d*w)/W
    cvzone.putTextRect(imgContour, f'uzaklik:{int(d)}cm', (x, y), scale=2, thickness=3,  # for calculating distance
                       colorT=(255, 255, 255), colorR=(0, 0, 0))

def stackImages(scale,imgArray):#birden fazla resmi birlestirme
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getcontours(img):  # sekilleri tutar
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:  # her bolgeyi bulma islemi
        area = cv2.contourArea(cnt)
        # print(area)
        if area > 500:  # eger bolge sadece 500 den buyuk ise cizmesi gerektigini soyledik (trashold yaptik )
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            # print(len(approx))#sekillerin kenar sayisini soyler 3 ucgen 4 kare 4 den fazla ise daire elde etmis oluruz
            objcor = len(approx)
            x, y, w, h = cv2.boundingRect(
                approx)  # tanimlanan sekillere yesil renk kare ekleyerek tanimlandiklarini belirliyoruz

            if objcor == 3:
                ObjectType = "triangle"
            elif objcor == 4:
                aspratio = w / float(h)
                if aspratio > 0.95 and aspratio < 1.05:
                    ObjectType = "SQUARE"  # kare olup olmadiginin tespiti
                else:
                    ObjectType = "Rectangle"
            elif objcor > 4:
                ObjectType = "circle"
            else:
                ObjectType = "none"
            if objcor>4:
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)   #yesil kare ile cevirmek
                cv2.putText(imgContour, ObjectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 2)
                drawDistance(w,x,y)

while True:
    ret,img=cap.read()
    if not ret:break


    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 10)  # kenarlari bulabilmek icin canny kullaniyoruz
    getcontours(imgCanny)
    imgblank = np.zeros_like(img)
    imgstack = stackImages(0.3, ([img, imgGray, imgBlur], [imgBlur, imgCanny, imgContour]))

    #########fps calculation
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(imgContour, fps, (7, 70), font, 3, (0, 255, 0), 3, cv2.LINE_AA)

    # Capture frame-by-frame
    if 0xFF == ord('q'): break
    cv2.imshow("imgBlur", imgContour)
    cv2.waitKey(1)









"""

#eski


import cv2
import cvzone
import numpy as np
from numba import jit, cuda
import time



cap=cv2.VideoCapture("videolar/testvideo.mp4")
prevCircle=None
dist=lambda x1,y1,x2,y2:(x1-x2)**2+(y1-y2)**2


# used to record the time when we processed last frame(for calculating fps)
prev_frame_time = 0
# used to record the time at which we processed current frame(for calculating fps)
new_frame_time = 0


def drawDistance(radius):
    w=radius#small w = weight in pixels
    W = 6.3#big W =weight in real life
    f = 750
    d = (W * f) / w  # calculated distance
    cvzone.putTextRect(frame, f'uzaklik:{int(d)}cm', (chosen[0] - 75, chosen[1] - 50), scale=2, thickness=3,colorT=(255, 255, 255), colorR=(0, 0, 0))
    #print(d)


while True:
    ret,frame=cap.read()
    if not ret:break
    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurFrame=cv2.GaussianBlur(grayFrame,(11,11),0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), font, 3, (0,255,0), 3, cv2.LINE_AA)

    ##########fps calculation


    circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=75, maxRadius=500)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen=None
        for i in circles[0,:]:
            if chosen is None:chosen=i
            if prevCircle is not None:
                if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1])<=dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                    chosen=i

        drawDistance(chosen[2])
        cv2.circle(frame,(chosen[0],chosen[1]),1,(0,100,100),3)
        cv2.circle(frame, (chosen[0], chosen[1]),chosen[2],(255,0,255),3)
        prevCircle=chosen
    cv2.imshow("test", frame)

    if cv2.waitKey(1)&0xFF ==ord('q'):break
cap.release()
cv2.destroyAllWindows()
"""


"""
"""
