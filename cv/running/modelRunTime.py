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
logger = logging.getLogger("modelRunTime")

class modelRunTime:
    def __init__(self, configs, device) -> None:
        self.device = device 
        self.configs = configs
        self.debug = configs['debugs']['showInfResults']

        if device == "tpu":
            from edgetpumodel import EdgeTPUModel
            import numpy as np
            weightsFile = configs['training']['weightsFile_tpu']
        else:
            from ultralytics import YOLO
            weightsFile = configs['training']['weightsFile']

        modelPath = Path(configs['training']['weightsDir'])
        modelFile = modelPath/weightsFile
        logger.info(f"model: {modelFile}")

        # Load the state dict
        if device == "tpu":
            thresh = min(configs['runTime']['distSettings']['handThreshold'],
                         configs['runTime']['distSettings']['objectThreshold'])
            self.model = EdgeTPUModel(modelFile, configs['training']['dataSet'], 
                                      conf_thresh=thresh, #only over this
                                      iou_thresh=configs['runTime']['distSettings']['nmsIouThreshold'],
                                      filter_classes=None,  # Not implemented
                                      agnostic_nms=False,
                                      max_det=100,
                                      v8=True)
            # Prime with the image size
            self.input_size = self.model.get_image_size()
            x = (255*np.random.random((3,*self.input_size))).astype(np.int8)
            self.model.forward(x) 
        else:
            self.model = YOLO(modelFile)  # 

    def runInference(self, image):
        #logger.info(f"image type: {type(image)}")

        if self.device == 'tpu':
            if isinstance(image, str):
                yoloResults =self.runInferenceTPUFile(image)
            else:
                yoloResults =self.runInferenceTPUWebCam(image)

            #inference time, nms time
            logger.info(f"TPU Inference, nms time: {self.model.get_last_inference_time()}") 
            #logger.info(f"TPU Results: {type(yoloResults)}, {yoloResults}")
            return yoloResults

        else:
            yoloResults = self.model.predict(image) # Returns a dict
            logger.info(yoloResults[0].speed)

            #logger.info(f"Results.boxes: {type(yoloResults[0].boxes.data)}, {yoloResults[0].boxes.data}")
            return yoloResults[0].boxes.data

    def runInferenceTPUFile(self, image):
            logger.info(f"Running TPU file infernece")
            # Returns a numpy array: x1, x2, y1, y2, conf, class
            results = self.model.predict(image, save_img=self.debug, save_txt=self.debug)

            return results

    def runInferenceTPUWebCam(self, image):
            from utils import get_image_tensor
            logger.info(f"Running TPU webcam infernece")
            full_image, net_image, pad = get_image_tensor(image, self.input_size[0])
            pred = self.model.forward(net_image)
            results = self.model.process_predictions(det=pred[0], 
                                                     output_image=full_image, 
                                                     pad=pad,
                                                     save_img=False,
                                                     save_txt=False,
                                                     hide_labels=True,
                                                     hide_conf=True)
                        
            #tinference, tnms = self.model.get_last_inference_time()
            #logger.info("Frame done in {}".format(tinference+tnms))
            return results
