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

#TODO: Add object name to display
#TODO: Add infernece probability to display

class displayHandObject:
    def __init__(self, handColor, objectColor, lineColor) -> None:
        self.handColor = handColor
        self.objectColor = objectColor
        self.lineColor = lineColor

    def draw(self, imgFile, dist):
        thisImg =  cv2.imread(imgFile)
        print(f"Image File shape: {imgFile} {thisImg.shape}")

        # The Object
        objUL, objLR = dist.getBox(dist.grabObject)
        cv2.rectangle(img=thisImg, pt1=objUL, pt2=objLR, color=self.objectColor, thickness=1)
        cv2.circle(img=thisImg, center=dist.bestCenter, radius=5, color=self.objectColor, thickness=2) #BGR

        # The Hand
        handUL, handLR = dist.getBox(dist.handObject)
        cv2.rectangle(img=thisImg, pt1=handUL, pt2=handLR, color=self.handColor, thickness=1)
        cv2.circle(img=thisImg, center=dist.handCenter, radius=5, color=self.handColor, thickness=2) #BGR

        # The distance
        label =f"Distance: {dist.bestDist:.0f}mm"
        print(label)
        cv2.putText(img=thisImg, text=label, org=objUL, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=self.lineColor)
        cv2.line(img=thisImg, pt1=dist.bestCenter, pt2=dist.handCenter, color=self.lineColor, thickness=1)

        cv2.imshow("", thisImg)
        cv2.waitKey()