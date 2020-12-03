import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

img_color = cv2.imread(r'F:\OpenCV\projects\sample1.jpg')
img = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
hImg , wImg = img.shape
conf = r'--oem 3 --psm 6 outputbase digits'
# print(pytesseract.image_to_string(img))
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split()
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img_color,(x,hImg-y),(w,hImg-h),(0,0,255),4)
#     cv2.putText(img_color,b[0],(x,hImg-y+30),cv2.FONT_HERSHEY_COMPLEX,1,(100,100,100),2)

boxess = pytesseract.image_to_data(img,config=conf)
for x,b in enumerate(boxess.splitlines()):
    if x!= 0 :
        b = b.split()
        if len(b) == 12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img_color,(x,y),(w+x,h+y),(0,0,255),4)
            cv2.putText(img_color,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(100,100,100),2)
cv2.imshow('imagee',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
