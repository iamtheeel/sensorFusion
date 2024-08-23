# sensorFusion, Computer Vision subsystem

The computer vision system is runing YOLOv5 on a [Corel Dev Board](https://coral.ai/products/dev-board) with a [Google Tensor Processing Unit (TPU)](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)

The computer vision consists of two catagories:
- [Training and Exporting](./training) the model
  - The training is written around [Ultralitics](https://docs.ultralytics.com/modes/train/)
- [Running](./running) the model
  - Run time is written around [jveitchmichaelis' edgetpu-yolo](https://github.com/jveitchmichaelis/edgetpu-yolo)

The images are captured with a generic webcam USB webcam

The output is displayed via a DVI monitor connected to the Corel board
