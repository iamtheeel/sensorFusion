# sensorFusion, Computer Vision: Running

The software will detect if you are running with a Cuda GPU, CPU, or MPS (Apple Silicon)

If you are running GPU or MPS, the inference, pre and postprocessing will be done with [Ultralitics](https://docs.ultralytics.com/modes/train/)

However the assumption is we will want to run with the TPU, running on a desktop/laptop/server is mostly for debuging.

If you are running on a TPU board the inference will be done with TensorFlow Lite, pre and postprocessing via [jveitchmichaelis' edgetpu-yolo](https://github.com/jveitchmichaelis/edgetpu-yolo)
 
[Documentation on preparing and using the Corel Dev Board](https://coral.ai/docs/dev-board/get-started) is fairly compleate and easy to follow.
I will present a distilled version with some notes.
I was installing from a laptop running Deebian Linux, and from OSX.

<br>

Serial Terminal:
> The [serial console](https://coral.ai/docs/dev-board/serial-console/ "Corel Dev Board Serial Console") for the dev-board is unusual:<br>
> <img src="readmeFiles/devboard-serial-console.jpg" height=300 alt="Serial connection Micro-USB"> <br>
> - The console port is the Micro-USB port on the same side as the 20 pin IO Pin Headder
> - The serial console power is sent via the USB-Micro-B connection, not via the main power.
>   - The TX/RX lights will iluminate on Console port power, then turn off when the board powers
> - The serial will be availalbe to the host prior to powering the board. It will, of course, not trasmit any data untill the board is powered.
> - It is recommended to connect to the serial console prior to powering the board
> - The serial settings are (115200, 8-N-1):
>   - Speed: 115200
>   - Data Bits: 8
>   - Parity Bit: None
>   - Stop Bits: 1
> - Default Credentials
>   - Username: mendel
>   - Password: mendel
>1. [Make sure your account on your host has permision to use the serial port. ](https://coral.ai/docs/dev-board/serial-console/ "Corel Dev Board Serial Console")
>1. Install "Screen" (or other terminal software, Note: on windows PuTTY is nice)
>1. Connect the Micro-USB to the host computer usb port.
>1. Determine the USB port (e.x. /dev/ttyUSB0)
>  - On Linux:
>    1. dmesg | grep ttyUSB
>    1. Select the first entry
>  - On OSX:
>    1. It will be: /dev/cu.SLAB_USBtoUART
>  - On Windows:
>    1. Install the driver (It should auto install when plugging the cable in)
>    1. Open the Device Manager (assuming MS has not moved it)
>    1. Look under "Ports (COM & LPT)" for Silicon Labs Dual CP2105 USB to UART Bridge"
>    2. The one that says somthing like: "Enhanced COM Port" (such as "COM3")"
>1. Connect to the board:
>   - screen /dev/ttyUSB0 115200
>   - To exit screen: <CTRL>+A, K, Y

<br>

Powering the board:
>- ** DO NOT POWER FROM YOUR COMPUTER** The board draws 2-3A, this may bork your computer's usb-c connection
>- Use an external powersupply that can source 3A at 5V (some laptop supplys are plenty good at 20V, but don't have much or any at 5V)
>- Use a usb-c cable that is good for 3A
>- There are 2 USB-C connections, one is for data (next to the DVI), the other is power (next to the audio):<br> <img src="readmeFiles/devboard-power-co.jpg" height=300 alt="Power via usbc">
>- Note: the red LED that comes on with power. The fan will also spin, and the serial port should start talking.



<br>

Install the TPU OS:
> - The operating system is out of date. As of this writting (Aug 2024), the newest version of [mendel linux](https://coral.ai/software/#mendel-linux) is from Nov 2021
> - Unlike a raspberry Pi, the OS must be installed on the internal memory, the computer can not boot from the microSD card (or at least not easily)
> - [There is a way](https://coral.ai/docs/dev-board/reflash/#flash-a-new-board) of flashing via USB, however I was not able to get it to work. Its possible that I had some silly error. But I did not spend too much time as I have a functioning SD Card. This also will not work for a fresh install, but should presev /home
>1. Download [Mendel Linux](https://coral.ai/software/#mendel-linux)
>  - Latest flashcard [enterprise-eagle-flashcard-20211117215217.zip](https://dl.google.com/coral/mendel/enterprise/enterprise-eagle-flashcard-20211117215217.zip)
>1. Put the image on a microSD card
>   1. Unzip the file
>   1. Determine which file is your SD Card
>      - df -l
>        - "df = disk format (asking not telling), -l = only show local disks
>      - This will give a list of mount points. If there is any doubt, unmount and eject the card and run the df again, looking for changes
>      - If your disk is not already formated for your OS, a df will not show. You can either use your os to format it, or go down a rabbit hole... Up to you.
>   1. Use dd to write the image e.x.:
>     - sudo dd if=flashcard_arm64.img of=/dev/disk7
>        - sudo = do as root
>        - dd = convert and copy a file
>        - if = infile
>        - of = outfile
>     - NOTE: this will overwrite whatever disk is pointed to with "of=<diskname>" by overwrite, I mean destroy. So, be durn spanky sure you have the correct disk.  
>1. Insert the SD Card to the Dev-Board card slot
> 1. Set the TPU to boot from SD
>    - The Dev board boot target is selected via the dip switches (Internal Memory Boot mode show):
> <br> <img src="readmeFiles/devboard-bootmode-emmc.jpg" height=300 alt="Boot Target Selection">
>6. Install the OS to internal memory
>  1. Connect to the serial consol (and/or plug a monitor in)
>  2. Power the board (this takes a while)
>  3. When the board says "Power Down" It is done
>1. Unplug the serial port and power (the serial is probably over kill, but why kill when you can overkill)
> 8. Set the TPU to boot from internal memory
> 1. Connect to the serial consol (and/or plug a monitor, keyboard, and mouse in)
> 10. Re-Power the bopard
>   - When boot is compleate the serial terminal will present the Mendel Linux logon prompt (the monitor will show the gui)

Dip Switch Configuration:
| Boot Mode | 1 | 2 | 3 | 4 |
|:-----------|---|---|---|---|
| SD Card | ON | OFF | ON | ON |
| Internal Memory | ON | OFF | OFF | OFF |

<br>

Configuring and operating the Corel TPU devboard:
> - Network
>   - The "easy" way to get the network setup is with nmtui
>   - This can be done with comandline arguments, or with the consol tool, we will use the tool
>   1. from a shell run: nmtui
>   2. Select "Activate a connection
>      - You should see a list of available coections
>      - If is does not show the list arrow down, it should refesh...
>   1. Select your Network <activate>
>     - Key in your password
> - SSH
> - KVM
> - X11

<br>

Software Installation:
This is a 3 year OS... It will work just fine for us, but give deprication warnings. We do have to be carfull about library useage tho.
>1. Install base code (if you have not already:
>    1. git clone https://github.com/iamtheeel/sensorFusion.git
>       - The required parts of edgetpu-yolo are included. I have modifyed very littel. But have made a few changes that were nessisary.
>1. Change directory to sensorFusion/cv/running
>1. install prerequisites:
>    - If on the TPU: Most of the following will give warnings (e.x.):
>    - script pip3.10 is installed in '/home/mendel/.local/bin' which is not on PATH.
>       - Fixable, but not a problem
>       - Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
>         - Fixable, but not a problem
>       - DEPRECATION: reportbug 7.5.3-deb10u1 has a non-standard version number. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of reportbug or contact the author to suggest that they release a version with a conforming version number. Discussion can be found at https://github.com/pypa/pip/issues/12063
>         - Y
>     1. sudo apt-get install -y python3 python3-pip
>     1. pip3 install --upgrade pip setuptools wheel
>     1. pip install -r --user requirements_tpu.txt
>   - Else (we need ultralitics):
>       - pip install -r requirements.txt


Running:
>1.



Setting up as a WIFI Host:
>1.

