# sensorFusion, Computer Vision: Training

Training and exporting are written around the [Ultralitics](https://docs.ultralytics.com/modes/train/) YOLO implementation

Installation:
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git\
>1. Change directory to training
>    1. cd sensorFusion/cv/training
>1. install prerequisites:
>    1. pip install -r requirements.txt
>1. Download Base Model from [Ultralytics](https://docs.ultralytics.com/models/yolov5/#supported-tasks-and-modes)
>    - NOTE: yolov5nu.py has been included.
>    1. Place the model in the "weights" directory.
>    - The weights have been trained on the COCO dataset.
>    - YOLOV5 is the [fastest/smallest](https://pytorch.org/hub/ultralytics_yolov5/) of the YOLO group
>    - The "Nano" [YOLOV5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5nu.pt) at 640px images size gives us an iference time of about 0.18s, good for 5Hz
>1. Download the model config file [yolov5.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml):
>    - Note: The modifyed file is part of the git hub
>    1. Place the file into the "models" directory
>    2. Modify the file to include 81 classes (our extra "hand" class)
>        - was> nc: 80 # number of classes 
>        - is > nc: 81 # number of classes
>3. Acquire training image set:
>    - The training set is too large to store on github. Contact MIC Lab for the dataset
>    - There is a utility set for converting from SSD style labels to YOLO style labels
>        - convertDataSet.py
>        - getLab.py
>    - There is a utility for viewing the dataset/labels:
>        - viewer.py  (also calls getLab.py

Configurations:

Running the Training:

Export the model:

Challenges: 
- As of 24-07-07, ultralitics export did not function with python revisions above 3.11
- Got an error on: from ultralytics import YOLO
  - Fix:        pip install -U --force-reinstall charset-normalizer
- On some systems: libGL error pops up: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
  - sudo apt update && sudo apt upgrade && sudo apt install build-essential
  - sudo apt-get install -y libgl1-mesa-dev
  - sudo apt-get install -y libglib2.0-0
- On some systems: a libusb error pops up: ImportError: libusb-1.0.so.0: cannot open shared object file: No such file or directory
  - sudo apt-get install -y libusb-1.0-0
