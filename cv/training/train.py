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
#model = YOLO("models/yolov3-tiny.yaml")  # build a new model from YAML
#model = YOLO("models/yolov5-p6n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov6n.yaml")  # build a new model from YAML
yoloModel = YOLO("models/yolov8n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8-p6n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("models/yolov8n.yaml").load("models/yolov8n.pt")  # build from YAML and transfer weights

yoloModel.info(detailed=True)

modelSum = summary(model=yoloModel.model, 
                   #mode=eval
                   #verbose=2
                   input_size=(1, image_depth, image_width, image_width), # make sure this is "input_size", not "input_shape"
            #col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "trainable"], 
            col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
            #col_width=20,
            row_settings=["var_names"]
            )


# image format

results = yoloModel.train(data="datasets/coco8.yaml", epochs=1, imgsz=image_width, device="mps")

