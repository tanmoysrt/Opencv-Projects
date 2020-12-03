import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True :
    res, frame =  cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    code = decode(gray)
    for i in code:
        myData  = i.data.decode('utf-8')
        size = i.rect
        cv2.rectangle(frame,(size[0]-4,size[1]-4),(size[0]+size[2]+4,size[1]+size[3]+4),(0,0,255),2)
        cv2.putText(frame,myData,(size[0]-10,size[1]-20),cv2.FONT_HERSHEY_COMPLEX,.5,(255,0,0),2)
    cv2.imshow("Image",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()