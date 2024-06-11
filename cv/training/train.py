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

# Training data
image_depth = 3
image_width = 96
image_height = image_width

# Load a model
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/
model = YOLO("models/yolov3-tiny.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("models/yolov8n.yaml").load("models/yolov8n.pt")  # build from YAML and transfer weights

model.info()

'''
modelSum = summary(model=model, 
            input_size=(1, image_depth, image_width, image_height), # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )
'''