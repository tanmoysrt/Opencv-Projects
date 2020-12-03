import cv2
import numpy as np 

cap = cv2.VideoCapture('http://192.168.1.101:4747/video')
minArea = 1000
filter = 0
while True:
    res,frame = cap.read()
    frame=frame[20:,:]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame,(5,5),1)
    edges = cv2.Canny(blur,40,80)
    kernel = np.ones((5,5))
    imgDial= cv2.dilate(edges,kernel,iterations=3)
    imgThres = cv2.erode(imgDial,kernel,iterations=2)
    contours,hierarchy=cv2.findContours(imgThres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    fianlContours=[]
    for i in contours:
        area= cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)
            approx= cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    fianlContours.append([len(approx),area,approx,bbox,i])
            else:
                fianlContours.append([len(approx),area,approx,bbox,i])
    fianlContours=sorted(fianlContours,key=lambda x:x[1],reverse=True)
    for i in fianlContours:
        cv2.drawContours(frame,i[4],-1,(0,0,255),3)

    cv2.imshow("Frame",frame)
    cv2.imshow("Thres",imgThres)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break


cap.release()
cv2.destroyAllWindows()