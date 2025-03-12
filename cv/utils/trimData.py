#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Trim the data set down
#
###
import os, fnmatch 
from pathlib import Path
import shutil

fixThis = '*_1_clip_*.jpg'
dataCapRate = 10
desiCapRate = 5
cutRatio = round(dataCapRate/desiCapRate)
cutThis = cutRatio

dataDir = "../images/day1"
destDir = Path(f"{dataDir}/arch")
destDir.mkdir(parents=True, exist_ok=True)
listing = os.scandir(dataDir)

for file in listing:
    if fnmatch.fnmatch(file, fixThis):
        imageFile_str =dataDir+'/'+file.name
        if cutThis == 1:
            cutThis = cutRatio
            #print(f"{cutThis}, {imageFile_str} keep")
        else:
            shutil.move(imageFile_str, destDir)
            #print(f"{cutThis}, {imageFile_str} CUT")

        cutThis = cutThis - 1