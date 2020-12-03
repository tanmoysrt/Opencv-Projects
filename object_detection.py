import cv2


# img = cv2.imread('lena.png')
cap = cv2.VideoCapture(0)
classnames = []
with open('projects/coco.names','rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

configPath = 'projects/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'projects/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True :
    ret, frame = cap.read()
    classIds, confs, bboxs = net.detect(frame,confThreshold = 0.5)
    # print(classIds)
    # print(bboxs)
    try:
        for classId, conf, bbox in zip(classIds.flatten(),confs.flatten(),bboxs):
            print(classId)
            print(bbox)
            cv2.rectangle(frame,bbox,color=(0,255,0),thickness=2)
            cv2.putText(frame,classnames[classId-1],(bbox[0]+10,bbox[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    except:
        print("ERROR")
    cv2.imshow("frame",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

# cv2.imshow("Image",img)
# cv2.waitKey(0)
cv2.destroyAllWindows()