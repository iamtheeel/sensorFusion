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

device = "cpu" 
if cuda.is_available(): device = "cuda" 
if backends.mps.is_available() and backends.mps.is_built(): device = "mps"

print(f"setup: Device = {device}")

epochs = 1
# Training data
image_depth = 3
#image_sz = 320 # Works
#image_sz = 240: Failes
#image_sz = 160 #works
#image_sz = 96 #works

#dataSet = "datasets/combinedData.yaml" # 2 class
dataSet = "datasets/coco_withHand.yaml" # coco 80 class, plus hand
#Dataset 'datasets/coco_withHand.yaml' images not found ⚠️, missing path '/Users/theeel/Documents/school/MIC/sensorFusion/src/cv/datasets/coco_withHand/images/val'
#dataSet = "datasets/foo.yaml"  # Single image set, not working
#dataSet = "datasets/dataset_ver1.yaml" # small(er) training set
#dataSet = "datasets/coco8.yaml"

# Load a model
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/
#yoloModel = YOLO("models/yolov3.yaml")  #  Does not work
#yoloModel = YOLO("models/yolov3-tiny.yaml")  # 
#yoloModel = YOLO("models/yolov5-p6n.yaml")  # 
#yoloModel = YOLO("models/yolov8n.yaml")  # 
#yoloModel = YOLO("models/yolov8-p6n.yaml")  # 
#yoloModel = YOLO("weights/yolov5nu.pt")  # trained with 300 epochs
yoloModel = YOLO("models/yolov5n.yaml").load("weights/yolov8n.pt")  # build from YAML and transfer weights
image_sz = 640 # Was trained with
freezeLayer = 10 # First 10 layers are the backbone (10: freezes 0-9)

# From: https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/#freeze-backbone
freeze = [f"model.{x}." for x in range(freezeLayer)]  # which layers to freeze
print(f"Layers to freeze: {freeze}")
print("--------------------------------------")
for k, v in yoloModel.named_parameters():
    #print(f"k: {k}")
    v.requires_grad = True
    if any(x in k for x in freeze):
        print(f"Freezing layer {k}")
        v.requires_grad = False

yoloModel.info(detailed=True)

modelSum = summary(model=yoloModel.model, 
                   #verbose=2,
                   input_size=(1, image_depth, image_sz, image_sz), # make sure this is "input_size", not "input_shape": must be square
            #col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "trainable"], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
            #col_width=20,
            row_settings=["var_names"]
            )

results = yoloModel.train(data=dataSet, epochs=epochs, imgsz=image_sz, device=device) # cpu, cuda, mps

exit()
