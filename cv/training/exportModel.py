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

def saveModel(modelDir, model, dataSet, imgH, imgW ):
    name = "best"
    modelPath = Path(modelDir)
    fileName = name+".pt"
    modelFile = modelPath/fileName

    # Load the state dict
    model.load_state_dict(torch.load(modelFile), strict=False) 

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
    #model.export(format="tflite", data=dataSet, int8=True) #Img is H, w
    model.export(format="tflite", data=dataSet, imgsz=(imgH, imgW), int8=True) #Img is H, w
    #model.export(format="tflite", imgsz=(imgMax, imgMax), int8=True, data=dataSet, optimize=True) # 
    #model.export(format="saved_model", imgsz=(imgMax, imgMax) ) # save as a tensorflow model

'''
    converter = tensorflow.lite.TFLiteConverter.from_saved_model(tfFile)
    # Some settings for the tfLite from MIC
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tensorflow.float32  
    #converter.inference_input_type = tensorflow.uint8  # this is giving headach
    converter.inference_output_type = tensorflow.float32
'''

    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i yolov8n_full_integer_quant.tflite.tflite > yoloV8n_int8.h

model = YOLO("models/yolov3-tiny.yaml" )  # build a new model from YAML
#model = YOLO("models/yolov8n.yaml" )  # build a new model from YAML
weightsDir = "runs/detect/train44/weights/" 
#weightsDir = "runs/detect/train21/weights/" 
#dataSet = "coco8.yaml"
#dataSet = "datasets/coco8.yaml"
dataSet = "datasets/dataset_ver1.yaml"

saveModel(weightsDir, model, dataSet, 96, 96) #TODO image size is messed up, reqirez square