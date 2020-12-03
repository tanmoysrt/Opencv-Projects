import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('http://192.168.1.100:4747/video')
screenCnt=[]
imgh = 480
imgw = 640

dst = np.array([
	[0, 0],
	[imgw - 1, 0],
	[imgw - 1, imgh - 1],
	[0, imgh - 1]], dtype = "float32")


def orderPoints(data):

    data = data.reshape((4,2))
    newdata = np.zeros((4,2),dtype=np.int32)
    add = data.sum(axis=1)
# Top-left || Bottom Right
    newdata[0]= data[np.argmin(add)]
    newdata[2]=data[np.argmax(add)]
    diff = np.diff(data, axis=1)
# Top Right || Bottom Left
    newdata[1]=data[np.argmin(diff)]
    newdata[3]=data[np.argmax(diff)]

    return newdata



while True :
    res, frame = cap.read()
    frame=frame[20:,:]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame,(5,5),0)
    edges = cv2.Canny(blur,30,80)
    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt != []:
        # print(screenCnt)
        print(orderPoints(screenCnt))
        M = cv2.getPerspectiveTransform(np.float32(orderPoints(screenCnt)),dst)
        warp = cv2.warpPerspective(frame,M,(imgw,imgh))
        cv2.drawContours(frame,[screenCnt],-1,(0,255,0),2)        
        cv2.imshow("wrap",warp)

        imgGrayFinal = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
        imgAdaptive = cv2.adaptiveThreshold(imgGrayFinal,255,1,1,7,2)
        imgAdaptive = cv2.bitwise_not(imgAdaptive)
        imgAdaptive = cv2.medianBlur(imgAdaptive,3)

        cv2.imshow("Adaptive",imgAdaptive)
    # try:
    #     c = max(contours,key=cv2.contourArea)
    #     x,y,w,h = cv2.boundingRect(c)
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     img2  = cv2.drawContours(frame,c,-1,(0,255,0),2)
    #     # cv2.imshow("img2",img2)
    # except ValueError:
    #     print("Not found")

    
    cv2.imshow("original",frame)
    # cv2.imshow("edges",edges)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()

