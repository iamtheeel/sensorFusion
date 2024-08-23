# sensorFusion, Computer Vision: Running

The software will detect if you are running with a Cuda GPU, CPU, or MPS (Apple Silicon)
If you are running GPU or MPS, the inference, pre and postprocessing will be done with [Ultralitics](https://docs.ultralytics.com/modes/train/)
If you are running on a TPU board the inference will be done with TensorFlow Lite, pre and postprocessing via [jveitchmichaelis' edgetpu-yolo](https://github.com/jveitchmichaelis/edgetpu-yolo)
 

Powering the board:

Serial Terminal:

Install the TPU OS:

Configuring and operating the Corel TPU devboard:

Software Installation:
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git
>1. Change directory to running
>1. install prerequisites:
>   - If on the TPU:
>       - pip install -r requirements_tpu.txt
>   _ Else:
>       - pip install -r requirements.txt
>              
