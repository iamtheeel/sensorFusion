#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# View the results
#
###
import cv2


#TODO: Change label position if the location obstructs it
#TODO: thumb throuh image if not ok

class displayHandObject:
    def __init__(self, handColor, objectColor, lineColor) -> None:
        self.handColor = handColor
        self.objectColor = objectColor
        self.lineColor = lineColor
        self.fullScreen = False

    def draw(self, imgFile, dist, valid):
        thisImg =  cv2.imread(imgFile)
        print(f"Image File shape: {imgFile} {thisImg.shape}")

        if dist.nHands != 0:
            self.drawHand(thisImg, dist)

        if dist.nNonHand != 0:
            self.drawObject(thisImg, dist)

        if valid:
            self.drawDistance(thisImg, dist)

        if self.fullScreen:
            cv2.namedWindow("sensorFusion", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("sensorFusion",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("sensorFusion", thisImg)

        waitkey = cv2.waitKey()
        return waitkey

    def drawObject(self, thisImg, dist):
        # The Object
        objText = f"Target: {dist.grabObject[4]:.2f}"
        print(f"drawObject: {dist.grabObject}")
        objUL, objLR = dist.getBox(dist.grabObject)
        cv2.rectangle(img=thisImg, pt1=objUL, pt2=objLR, color=self.objectColor, thickness=1)
        cv2.circle(img=thisImg, center=dist.bestCenter, radius=5, color=self.objectColor, thickness=2) #BGR
        cv2.putText(img=thisImg, text=objText, org=objUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.objectColor)

    def drawHand(self, thisImg, dist):
        # The Hand
        handText = f"Hand: {dist.handObject[4]:.2f}"
        handUL, handLR = dist.getBox(dist.handObject)
        cv2.rectangle(img=thisImg, pt1=handUL, pt2=handLR, color=self.handColor, thickness=1)
        cv2.circle(img=thisImg, center=dist.handCenter, radius=5, color=self.handColor, thickness=2) #BGR
        cv2.putText(img=thisImg, text=handText, org=handUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=self.handColor)

    def drawDistance(self, thisImg, dist):
        # The distance
        distText =f"Distance: {dist.bestDist:.0f}mm"
        distUL =  (5,15)
        print(distText)
        cv2.line(img=thisImg, pt1=dist.bestCenter, pt2=dist.handCenter, color=self.lineColor, thickness=1)
        cv2.putText(img=thisImg, text=distText, org=distUL, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=self.lineColor)