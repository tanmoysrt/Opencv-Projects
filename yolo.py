import cv2
import numpy as np

cap = cv2.VideoCapture('http://192.168.1.101:4747/video')
classesFile = 'projects/yolo/coco.names'
configFile = 'projects/yolo/yolov3.cfg'
weightFile = 'projects/yolo/yolov3.weights'
confthres = 0.5
nms_threshold=0.3

net = cv2.dnn.readNetFromDarknet(configFile,weightFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


classesName =[]
with open(classesFile,'rt') as f:
    classesName = f.read().rstrip('\n').split('\n')


def findObjects(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for i in output:
            scores = i[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confthres :
                w,h = int(i[2]*wT),int(i[3]*hT)
                x,y = int(i[0]*wT) , int(i[1]*hT-hT/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confthres,nms_threshold)
    # print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classesName[classIds[i]]} {int(confs[i]*100)} %',(x+50,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

# print(classesName)
# print(len(classesName))

while True:
    res,frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame,1/255,(160,160),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames =  net.getLayerNames()

    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)    

    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs,frame)

    cv2.imshow("Original",frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()