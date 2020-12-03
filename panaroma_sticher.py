import cv2
import os

mainFolder = 'images'
myFolders = os.listdir(mainFolder)
print(myFolders)

count = 0
for folder in myFolders:
    path = mainFolder+'/'+folder
    images = []
    myList = os.listdir(path)

    for files in myList: 
        curImg = cv2.imread(f'{path}/{files}')
        curImg = cv2.resize(curImg,(0,0),None,0.2,0.2)
        images.append(curImg)

    sticher = cv2.Stitcher.create()
    (status,result)= sticher.stitch(images)
    if ( status == cv2.STITCHER_OK):
        print('succeeded')
        cv2.imshow("Result"+str(count),result)
    else:
        print("failed")
    count+=1


cv2.waitKey(0)
cv2.destroyAllWindows()