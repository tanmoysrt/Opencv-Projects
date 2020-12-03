import cv2
import math

img = cv2.imread('angle1.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
pointlist=[]

def mouse_clicks(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),2,(255,0,0),3)
        pointlist.append([x,y])
        print(pointlist)
        if len(pointlist)%3==0:
            print("HELLO ")
            getAngle()
def getAngle():
    a = pointlist[-1]
    b = pointlist[-2]
    c = pointlist[-3]
    try : 
        m1 = (b[1]-a[1])/(b[0]-a[0])
        m2 = (c[1]-b[1])/(c[0]-b[0])
        
        m12 = (m1-m2)/(1+m1*m2)
        angle = round(math.degrees(math.atan(m12)),1)
        if angle < 0 :
            angle = 180+angle
    except Exception as e :
        print(e)
        angle = "undefined"
    cv2.line(img,(a[0],a[1]),(b[0],b[1]),(255,0,0),4)
    cv2.line(img,(c[0],c[1]),(b[0],b[1]),(255,0,0),4)
    cv2.putText(img,str(angle)+" degree",(b[0]+20,b[1]-7),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
def absoluteValue(i):
    if i < 0 :
        return -i
    return i
while True:
    cv2.imshow("Image",img)
    cv2.setMouseCallback("Image",mouse_clicks)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointlist=[]
        img = cv2.imread('angle1.png') 
cv2.destroyAllWindows()