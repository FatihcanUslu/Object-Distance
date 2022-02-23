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

