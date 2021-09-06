import cv2
import numpy as np
import os
import HandTrackingModule as htm


brushThickness = 25
eraserThickness = 100

folderPath = "Header-files"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img, draw=False)

    # find coordinates of landmark
    lmList = detector.findPosition(img, draw=False)         
    
    if len(lmList) != 0:
    
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        fingers = detector.fingersUp()
        print(fingers)

        # check for selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            
            if y1 < 125:
                if 19 < x1 < 126:
                    if brushThickness>1:
                        brushThickness-=1           
                    cv2.putText(img, "Size: "+str(brushThickness), (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                elif 156 < x1 < 262:
                    if brushThickness<100:
                        brushThickness+=1
                    cv2.putText(img, "Size: "+str(brushThickness), (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)    
                elif 454 < x1 < 627:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 676 < x1 < 844:
                    header = overlayList[1]
                    drawColor = (255, 50, 0)
                elif 895 < x1 < 1060:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1096 < x1 < 1257:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        #  check for dwaing mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

        else:
            xp,yp = 0,0


        # Clear Canvas if YOO!
        if(fingers[0]==0 and fingers[1]==1 and fingers[2]==0 and fingers[3]==0 and fingers[4]==1):
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

