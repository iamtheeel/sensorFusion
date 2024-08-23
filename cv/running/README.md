# sensorFusion, Computer Vision: Running

The software will detect if you are running with a Cuda GPU, CPU, or MPS (Apple Silicon)

If you are running GPU or MPS, the inference, pre and postprocessing will be done with [Ultralitics](https://docs.ultralytics.com/modes/train/)

If you are running on a TPU board the inference will be done with TensorFlow Lite, pre and postprocessing via [jveitchmichaelis' edgetpu-yolo](https://github.com/jveitchmichaelis/edgetpu-yolo)
 
[Documentation on preparing and using the Corel Dev Board](https://coral.ai/docs/dev-board/get-started) is fairly compleate and easy to follow.
I will present a distilled version with some notes.
I was installing from a laptop running Deebian Linux

<br>

Powering the board:
>- ** DO NOT POWER FROM YOUR COMPUTER** The board draws 2-3A, this may bork your computer's usb-c connection
>- Use an external powersupply that can source 3A at 5V (some laptop supplys are plenty good at 20V, but don't have much or any at 5V)
>- Use a usb-c cable that is good for 3A
>- There are 2 USB-C connections, one is for data (next to the DVI), the other is power (next to the audio):<br> <img src="readmeFiles/devboard-power-co.jpg" height=300 alt="Power via usbc">
>- Note: the red LED that comes on with power. The fan will also spin, and the serial port should start talking.

<br>

Serial Terminal:

<br>

Install the TPU OS:
> - The operating system is out of date. As of this writting (Aug 2024), the newest version of [mendel linux](https://coral.ai/software/#mendel-linux) is from Nov 2021
> - Unlike a raspberry Pi, the OS must be installed on the internal memory, the computer can not boot from the microSD card (or at least not easily)
> - There is a [way](https://coral.ai/docs/dev-board/reflash/#flash-a-new-board) of flashing via USB, however I was not able to get it to work. Its possible that I had some silly error. But I did not spend too much time as I have a functioning SD Card.
>1. Download the boot image
>2. Put the image on a microSD card
>3. Set the TPU to boot from SD
>4. Install the OS to internal memory
>5. Set the TPU to boot from internal memory

<br>

Configuring and operating the Corel TPU devboard:

<br>

Software Installation:
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git
>1. Change directory to running
>1. install prerequisites:
>   - If on the TPU:
>       - pip install -r requirements_tpu.txt
>   _ Else (we need ultralitics):
>       - pip install -r requirements.txt
>              
