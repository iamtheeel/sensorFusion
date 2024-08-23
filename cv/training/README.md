# sensorFusion, Computer Vision: Training

Training and exporting are written around the [Ultralitics](https://docs.ultralytics.com/modes/train/) YOLO implementation

Instalation:
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git\
>1. Change directory to training
>    1. cd sensorFusion/cv/training
>1. install prerequisites:
>    1. pip install -r requirements.txt
>1. Download Base Model
>2. Acquire training image set:
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
