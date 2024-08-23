# sensorFusion, Computer Vision: Training

Training and exporting are written around the [Ultralitics](https://docs.ultralytics.com/modes/train/) YOLO implementation
We are using the YOLOv8 ultralytics engine to train a YOLOv5 model. This needs to be taken account for at run time. More on that there.

The final product is a 8bit (unsigned for YOLOV8) quantized tensorflow light model that is compiled on the TPU.

Installation:
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git\
>1. Change directory to training
>    1. cd sensorFusion/cv/training
>1. install prerequisites:
>    1. pip install -r requirements.txt
>1. Download Base Model from [Ultralytics](https://docs.ultralytics.com/models/yolov5/#supported-tasks-and-modes)
>    - **NOTE: yolov5nu.py is included in this repository**
>    1. Place the model in the "weights" directory.
>    - The weights have been trained on the COCO dataset.
>    - YOLOV5 is the [fastest/smallest](https://pytorch.org/hub/ultralytics_yolov5/) of the YOLO group
>    - The "Nano" [YOLOV5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5nu.pt) at 640px images size gives us an iference time of about 0.18s, good for 5Hz
>1. Download the model config file [yolov5.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml):
>    - **Note: The modifyed yolov5.yaml is included in this repository**
>    1. Place the file into the "models" directory
>    2. Modify the file to include 81 classes (our extra "hand" class)
>        - was> nc: 80 # number of classes 
>        - is > nc: 81 # number of classes
>3. Acquire training image set:
>    - the file [coco_withHand.yaml](../datasets/coco_withHand.yaml) contains:
>        - the location of the images and labels
>        - the class names (modifyed to include "hand")
>    - The training set is too large to store on github. Contact MIC Lab for the dataset
>    - There is a utility set for converting from SSD style labels to YOLO style labels
>        - convertDataSet.py
>        - getLab.py
>    - There is a utility for viewing the dataset/labels:
>        - viewer.py  (also calls getLab.py

<br>

Configurations:
>- The CV configureation file is [config.yaml](../config.yaml)
>    - The file is shared from training and runtime to minimize configuration mismatch issues
>- Training Configurations:
>    - debugs:
>        - debug: True
>            - Prints out possibly usefull information
>    - training:
>        - imageSize [height, width]: [480, 640]
>            - Note, most settings assume a square image, will select the larger of the two numbers
>        - dataSet:
>            - the location and name of the dataset configuration file
>            - set to [../datasets/coco_withHand.yaml](../datasets/coco_withHand.yaml)
>        - modelsDir:
>            - The location of the model file: "../models"
>        - modelFile:
>            - the YOLO model configuration: "yolov5nu.yaml"
>            - Notes:
>                - the "n" is for the nano size
>                - The "u" is to treat the model as V8... I think
>        - transLearn:
>            - Train from an existing pretrained weights file, or train from scratch: True
>        - weightsDir (only used if transLearn): 
>            - The location of the weights file: "../weights"
>        - weightsFile (only used if transLearn):
>            - The pre-trained weights file:  "yolov5nu.pt"
>        - epochs:
>            - The number of epochs to run for: 300
>        - freezeLayer (only used if transLearn):
>            - How many layers to freeze before learning: 10
>- Export Configurations:
>    - debugs:
>        - debug: True
>            - Prints out possibly usefull information
>    - training:
>        - imageSize [height, width]: [480, 640]
>            - Note, most settings assume a square image, will select the larger of the two numbers
>        - dataSet:
>            - the location and name of the dataset configuration file
>            - set to [../datasets/coco_withHand.yaml](../datasets/coco_withHand.yaml)1
>            - weightsDir (only used if transLearn): 
>            - The location of the weights file: "../weights"
>        - weightsFile (only used if transLearn):
>            - The weights file created in training:  "yolo5nu_hsv_h1.0_81classes.pt"


<br>

Running the Training:
> 1. Edit the Configuration file as needed
> 2. Change to the "training" directory
> 3. Run the script:
>     - python3 train.py
> 4. Wait
>     - A new run number will be created for each run. 
>     - The created weights file will be in:
>         - sensorFusion/cv/runs/train<run number>/weights/best.py
>     - Lots of fun data, logs and so on is in:
>         - sensorFusion/cv/runs/train<run number>
> 5. Copy the newly created weights file to ../weights
>     - Give it a sensible name that contains the info you may be playing with e.x.:
>         - yolo5nu_hsv_h1.0_81classes.pt
<br>

Export the model:
- **NOTE: This is a ram hog**, like 20mb. And the error is not obvious: See Challenges
> 1. Edit the Configuration file as needed
>     - E.x.: weightsFile: "yolo5nu_hsv_h1.0_81classes.pt"
> 3. Change to the "training" directory
> 4. Run the script:
> 5. Wait, like up to an hour, might look hung. It's not... probably
> 6. Output:
>     - Quantization calibration data from the training dataset will be created in the [run directory](.) e.x.:
>       - calibration_image_sample_data_20x128x128x3_float32.py
>     - The export script will create the output relative to the [weights directory](../weight) based on the weights file name:
>       - an ONNX file: yolo5nu_hsv_h1.0_81classes.onnx
>       - A directory of the export files: yolo5nu_hsv_h1.0_81classes including but not limited to:
>           - Full integer quantized tensorflow lite file: yolo5nu_hsv_h1.0_81classes_full_integer_quant.tflite
>           - An edge tpu compiled version of that file: yolo5nu_hsv_h1.0_81classes_full_integer_quant_edgetpu.tflite
>           - A log file of the TPU complile: yolo5nu_hsv_h1.0_81classes_full_integer_quant_edgetpu.log
>           - Note: the _int8.tflite file is a mixed quantization and can not be used here. We must use the full integer quanization
> 7. Copy the file to the tensorflow board (see doc in [running](../running)

<br>

Challenges: 
- As of 24-07-07, ultralitics export did not function with python revisions above 3.11
- Got an error on: from ultralytics import YOLO
  - Fix: pip install -U --force-reinstall charset-normalizer
- On some systems: libGL error pops up: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
  - sudo apt update && sudo apt upgrade && sudo apt install build-essential
  - sudo apt-get install -y libgl1-mesa-dev
  - sudo apt-get install -y libglib2.0-0
- On some systems: a libusb error pops up: ImportError: libusb-1.0.so.0: cannot open shared object file: No such file or directory
  - sudo apt-get install -y libusb-1.0-0
- Export is a ram hog and the error is not obvious
  - Error: "zsh: killed  /<..your python location..>/bin/python"
  - Exporting with a smaller imageSize can be a workaround, but if you need the image size you need more ram:
    
