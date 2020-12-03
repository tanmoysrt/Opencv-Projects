import cv2
import numpy as np
import utils

path = "1.jpg"


img = cv2.imread(path)
ans = [1,2,0,4,4]
# PREPROCESSING
img =  cv2.resize(img,(640,640))
imgFinal = img.copy()
imgContours = img.copy()
imgBiggestContour = img.copy()
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray,(5,5),1)
imgcany = cv2.Canny(imgblur,10,50)

# FIND CONTOUR
contours, hierarchy = cv2.findContours(imgcany,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
# FIND RECTANGLE
rectCountour=utils.rectContour(contours)
biggestCountour = utils.getCornerPoints(rectCountour[0])
gradePoints = utils.getCornerPoints(rectCountour[1])
# print(biggestCountour)

if biggestCountour.size != 0 and gradePoints.size != 0 :
    cv2.drawContours(imgBiggestContour,biggestCountour,-1,(0,255,0),10)
    cv2.drawContours(imgBiggestContour,gradePoints,-1,(255,0,0),10)
    biggestCountour=utils.reorder(biggestCountour)
    gradePoints=utils.reorder(gradePoints)

    pt1 = np.float32(biggestCountour)
    pt2 = np.float32([[0,0,],[640,0],[0,640],[640,640]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWrapColor = cv2.warpPerspective(img,matrix,(640,640))

    ptg1 = np.float32(gradePoints)
    ptg2 = np.float32([[0,0,],[325,0],[0,150],[325,150]])
    matrixg = cv2.getPerspectiveTransform(ptg1,ptg2)
    imggWrapColor = cv2.warpPerspective(img,matrixg,(325,150))
    # cv2.imshow("x",imggWrapColor)

    # THRESHOLD
    imgWrapGray = cv2.cvtColor(imgWrapColor,cv2.COLOR_BGR2GRAY)
    imgThreshx = cv2.threshold(imgWrapGray,180,255,cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThreshx)
    # cv2.imshow("h",boxes[1])
    # print(cv2.countNonZero(boxes[1]))

    myPixelVal = np.zeros((5,5))
    countC = 0 
    countR = 0 

    for image in boxes :
        totalpixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalpixels
        countC+=1
        if countC == 5 :
            countR +=1
            countC = 0
    # print(myPixelVal)

    myIndex = []
    
    for x in range(0,5):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)
    grading = []
    for x in range(0,5):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    # print(grading)
    score = np.sum(grading)/5*100
    # print(score)

    imgRes = imgWrapColor.copy()
    imgRes = utils.showAnswers(imgRes,myIndex,grading,ans,5,5)
    imgResraw = np.zeros_like(imgRes)
    imgResraw = utils.showAnswers(imgResraw,myIndex,grading,ans,5,5)
    invmatrix = cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWarp = cv2.warpPerspective(imgResraw, invmatrix, (640, 640))

    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    # cv2.imshow("fgfgf",imgFinal)

    imgRawGrade = np.zeros_like(imggWrapColor)
    cv2.putText(imgRawGrade,str(int(score))+"%",(65,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    invmatrixG = cv2.getPerspectiveTransform(ptg2,ptg1)
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invmatrixG, (640, 640))
    # cv2.imshow("FDFDF",imgInvGradeDisplay)
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)



imgBlank = np.zeros_like(img)
imgArray=([img,imggray,imgblur,imgcany],
[imgContours,imgBiggestContour,imgWrapColor,imgThreshx],
[imgRes,imgResraw,imgInvWarp,imgFinal])

imgstacked = utils.stackImages(imgArray,0.5)
imgstacked =  cv2.resize(imgstacked,(640,640))

cv2.imshow("Stacked Images",imgstacked)
cv2.imshow("Final Image",imgFinal)
cv2.waitKey(0)