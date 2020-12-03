import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('elonmask.JPG')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img2 = face_recognition.load_image_file('elonmask2.JPG')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

cv2.imshow("Elon MAsk",img)
cv2.imshow("Elon Test",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()