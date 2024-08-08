#
# Sensor Fusion
#
# MIC Lab
# Joshua B. Mehlman
# Summer 2024
#
###
#
# Save the model to TensorFlow MicroControler
# must run python 3.11
#
###
from pathlib import Path
from ultralytics import YOLO
import torch

# From MIC
def representative_dataset():
    import numpy as np
    """
    Prepare representive data for quantization activation layer
    """

    ### Make a function
    imgW = 128
    imgH = 128
    imgD = 3

    data = np.load("./calibration_image_sample_data_20x128x128x3_float32.npy",allow_pickle=True) #Created in DataPreperation
    print(f"Data size: {data.shape}") # Batch size: 87, 2, 96, 96
    for i in range(len(data)):
        temp_data = data[i]
        temp_data = temp_data.reshape(1,imgW,imgH,imgD)# 2 is from RGB565, 2 bytes for 3 colors
        #temp_data = temp_data.reshape(1,2,imgW,imgH)# 2 is from RGB565, 2 bytes for 3 colors
        yield [temp_data.astype(np.float32)]

def saveModel(modelFile, dataSet, imgH, imgW ):

    # Load the state dict
    #model.load_state_dict(torch.load(modelFile), strict=False) 
    model = YOLO(modelFile)  # build a new model from YAML

    # Save as ONNX
    #https://docs.ultralytics.com/modes/export/#arguments
    #model.export(format="onnx", imgsz=imgHeight, int8=True, , optimize=True)
    #onnxFileName = name+".onnx"
    #onnxFile = modelPath/onnxFileName
#    model.export(format="onnx",  imgsz=imgMax)
    #print(f"Saveing to Onnx: {onnxFile}")
    '''
    input_shape = (1, imgLayers, imgWidth, imgHeight)
    #exit()
    torch.onnx.export(model, torch.randn(input_shape), onnxFile, verbose=True,
                      input_names=["input"],
                      output_names=["output"])
    '''

    '''
    model.export(format="onnx",  imgsz=(imgMax, imgMax), dynamic=False)
    #onnxFile = "yolov8n" + ".onnx"

    # Convert to Tensorflow
    import onnx2tf
    onnxFile = "yolov3-tiny" + ".onnx"
    tfFile = "yolov3n"+"_tf"

    print(f"Saveing to Tensor FLow: {tfFile} with onnx2tf")
    #calibrationData=[ ["input", "../output/representive_data.npy", mean, std] ]
    onnx2tf.convert(
        input_onnx_file_path=onnxFile,
        output_folder_path=tfFile,
        #overwrite_input_shape=[1,3,96,96],
        output_nms_with_dynamic_tensor = True,
        #output_integer_quantized_tflite=True,
        #custom_input_op_name_np_data_path=calibrationData,
        #copy_onnx_input_output_names_to_tflite=True,
        #non_verbose=True,
    )
    '''

    # Convert to TF Lite
    # https://docs.ultralytics.com/modes/export/#export-formats
    # barking: ModuleNotFoundError: No module named 'imp'
    # Run with python < 3.12: https://docs.python.org/3.11/library/imp.html
    # But it barks AFTER it saves the tfLite
    # Has started hanging. No clue what changed. But it is generating to the representative_dataset
    #  Works on the linux box and the server. I think upgrade utralitics version

    '''
    must use python 3.11 for the exporter to work as of 7/7/24
    '''
    #model.export(format="tflite", data=dataSet, int8=True) 
    #model.export(format="tflite", data=dataSet, imgsz=(imgH, imgW), int8=True)
    # https://github.com/ultralytics/ultralytics/issues/1185  #I did not seem to have a problem tho
    model.export(format="edgetpu", data=dataSet, imgsz=(imgH, imgW), int8=True) #Edge TPU is linux only
    #Img is H, w
    #model.export(format="tflite", imgsz=(imgMax, imgMax), int8=True, data=dataSet, optimize=True) # 
    #model.export(format="saved_model", imgsz=(imgMax, imgMax) ) # save as a tensorflow model


    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i yolov8n_full_integer_quant.tflite.tflite > yoloV8n_int8.h


#weightsDir = "runs/detect/train9/weights/" #glass: YoloV3, imgsz=320
#weightsDir = "runs/detect/train10/weights/" #glass: YoloV8, imgsz=320
#weightsDir = "runs/detect/train11/weights/" #glass: YoloV8, imgsz=96
#weightsDir = "runs/detect/train29/weights/" #glass: YoloV3, imgsz=320, d,w: 
#modelDir = "runs/detect/train30/weights/" 
modelDir = "weights/" 
modelPath = Path(modelDir)
fileName = "yolov5nu_orig.pt"
#fileName = "best.pt"
modelFile = modelPath/fileName

#dataSet = "coco8.yaml"
#dataSet = "datasets/coco8.yaml"
#dataSet = "datasets/dataset_ver1.yaml"
dataSet = "datasets/combinedData.yaml"
imgH = 96
imgW = 96

saveModel(modelFile=modelFile, dataSet=dataSet, imgH=imgH, imgW=imgW) 