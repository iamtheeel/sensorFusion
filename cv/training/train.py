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

# Training data
image_depth = 3
image_width = 96
image_height = image_width

# Load a model
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/
#model = YOLO("models/yolov3-tiny.yaml")  # build a new model from YAML
#model = YOLO("models/yolov5-p6n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov6n.yaml")  # build a new model from YAML
model = YOLO("models/yolov8n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8-p6n.yaml")  # build a new model from YAML
#model = YOLO("models/yolov8n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("models/yolov8n.yaml").load("models/yolov8n.pt")  # build from YAML and transfer weights

model.info(detailed=True)

results = model.train(data="coco8.yaml", epochs=1, imgsz=image_width, device="mps")

