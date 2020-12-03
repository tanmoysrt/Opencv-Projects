import cv2
import numpy as np
from pyzbar.pyzbar import decode

# cap = cv2.VideoCapture(0)

# while True :
#     res, frame =  cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     code = decode(gray)
#     print(code)
#     k = cv2.waitKey(0) & 0xFF
#     if k == 27 :
#         break

# cap.release()
# cv2.destroyAllWindows()

img = cv2.imread('barcode1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

code=decode(gray)
for i in code:
    myData  = i.data.decode('utf-8')
    size = i.rect
    print(size[0])
    cv2.rectangle(img,(size[0]-4,size[1]-4),(size[0]+size[2]+4,size[1]+size[3]+4),(0,0,255),2)
    cv2.putText(img,myData,(size[0]-50,size[1]-20),cv2.FONT_HERSHEY_COMPLEX,.3,(255,0,0),1)
    print(myData)
cv2.imshow("Image",img)
cv2.waitKey(0)