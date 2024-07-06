#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Training for YOLO to find distance from Glove to object to grasp
#
###

from ultralytics import YOLO
from torchinfo import summary

from torch import cuda, backends

device = "cuda" if cuda.is_available() else "cpu"
if backends.mps.is_available():
    if backends.mps.is_built():
        device = "mps"

print(f"setup: Device = {device}")
exit()
# Training data
image_depth = 3
image_sz = 320 # Works
#image_sz = 240: Failes
#image_sz = 160 #works
#image_sz = 96 #works

dataSet = "datasets/combinedData.yaml"
#dataSet = "datasets/dataset_ver1.yaml"
#dataSet = "datasets/coco8.yaml"

# Load a model
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/
#yoloModel = YOLO("models/yolov3.yaml")  #  Does not work
yoloModel = YOLO("models/yolov3-tiny.yaml")  # 
#yoloModel = YOLO("models/yolov5-p6n.yaml")  # 
#yoloModel = YOLO("models/yolov8n.yaml")  # 
#yoloModel = YOLO("models/yolov8-p6n.yaml")  # 
#yoloModel = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)
#yoloModel = YOLO("models/yolov8n.yaml").load("models/yolov8n.pt")  # build from YAML and transfer weights

yoloModel.info(detailed=True)

modelSum = summary(model=yoloModel.model, 
                   #verbose=2,
                   input_size=(1, image_depth, image_sz, image_sz), # make sure this is "input_size", not "input_shape": must be square
            #col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "trainable"], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
            #col_width=20,
            row_settings=["var_names"]
            )

#exit()
# TODO image format
#WARNING ⚠️ updating to 'imgsz=96'. 'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'
results = yoloModel.train(data=dataSet, epochs=10, imgsz=image_sz, device=device) # cpu, cuda, mps

'''

TensorBoard: Start with 'tensorboard --logdir runs/detect/train36', view at http://localhost:6006/
Freezing layer 'model.22.dfl.conv.weight'

train: Scanning /Users/theeel/Documents/school/MIC/sensorFusion/src/cv/datasets/dataset_ver1/labels/train.cache... 304 images, 93 backgrounds, 0 cor
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 9, len(boxes) = 211. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
val: Scanning /Users/theeel/Documents/school/MIC/sensorFusion/src/cv/datasets/dataset_ver1/labels/val.cache... 36 images, 8 backgrounds, 0 corrupt: 
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 1, len(boxes) = 28. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
Plotting labels to runs/detect/train36/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001667, momentum=0.9) with parameter groups 53 weight(decay=0.0), 60 weight(decay=0.0005), 59 bias(decay=0.0)
TensorBoard: model graph visualization added ✅

'''