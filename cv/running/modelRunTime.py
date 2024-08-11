#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# sorts out if we are on a tpu, or not
# Sending camera data or saved image
# Sends the image to the appropriate inference model
#
###
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runModel")

class modelRunTime:
    def __init__(self, configs, imgSrc, debug, device) -> None:
        self.device = device 
        self.imgSrc = imgSrc
        self.configs = configs
        self.debug = debug

        if device == "tpu":
            from edgetpumodel import EdgeTPUModel
            import numpy as np
            weightsFile = configs['weightsFile_tpu']
        else:
            from ultralytics import YOLO
            weightsFile = configs['weightsFile']

        modelPath = Path(configs['weightsDir'])
        modelFile = modelPath/weightsFile
        logger.info(f"model: {modelFile}")

        # Load the state dict
        if device == "tpu":
            self.model = EdgeTPUModel(modelFile, configs['training']['dataSet'], conf_thresh=0.1, iou_thresh=0.1, v8=True)
            self.input_size = self.model.get_image_size()
            x = (255*np.random.random((3,*self.input_size))).astype(np.int8)
            self.model.forward(x) # Prime with the image size
        else:
            self.model = YOLO(modelFile)  # 

    def runInference(self, image):
        #logger.info(f"image type: {type(image)}")

        if self.device == 'tpu':
            if isinstance(image, str):
                results =self.runInferenceTPUFile
            else:
                results =self.runInferenceTPUWebCam

            logger.info(f"Inference, nms time: {self.model.get_last_inference_time()}") #inference time, nms time
            logger.info(f"Results: {type(results)}, {results}")

        else:
            yoloResults = self.model.predict(image) # Returns a dict
            logger.info(yoloResults[0].speed)

            #logger.info(f"Results.boxes: {type(yoloResults[0].boxes.data)}, {yoloResults[0].boxes.data}")

            results = yoloResults[0].boxes.data
        return results

    def runInferenceTPUFile(self, image):
            logger.info(f"Running TPU file infernece")
            # Returns a numpy array: x1, x2, y1, y2, conf, class
            results = self.model.predict(image, save_img=self.debug['showInfResults'] , save_txt=self.debug['showInfResults'])

            return results

    def runInferenceTPUWebCam(self, image):
            from utils import get_image_tensor
            logger.info(f"Running TPU webcam infernece")
            full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
            pred = self.model.forward(net_image)
            self.model.process_predictions(pred[0], full_image, pad)
                        
            tinference, tnms = self.model.get_last_inference_time()
            logger.info("Frame done in {}".format(tinference+tnms))