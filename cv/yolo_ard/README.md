Note: 
The Spresens Tensor Flow Build does not have logging information. Use the build from the Mobile and Intelligent Computing Laboratory http://sfsu-miclab.org/ <br>
https://github.com/miclab-sfsu/CNN-KeywordSpotting-On-Device-ML/blob/main/keywordspotting_sonyspresense/keywordspotting_sonyspresense.ino <br>
- Add the following to to Additional Boards manager https://github.com/zhenyulincs/Sony-Spresense-Arduino-TFMicro/releases/download/2.0/package_spresense_local_index.json
- Select the SFSU-MICLab SPRESENSE_Tensorflow board
- Librarys used:
  - Camera.h
  - SDHCI.h (The SD Card)
  - Tensorflow:
    - tensorflow/lite/micro/micro_mutable_op_resolver.h
    - tensorflow/lite/micro/micro_interpreter.h
    - tensorflow/lite/micro/system_setup.h
    - tensorflow/lite/micro/spresense/debug_log_callback.h
    - tensorflow/lite/schema/schema_generated.h
      
Settings:
- Tools, Memory, Set to Max (1,536 kB)
- Tools, Upload Speed, Set to Max (1,152,000). Not required, but 1.5MB takes a while to upload.

Serial Comms:
- 1,000,000 8-N-1. We want this FAST for when we stream BMPs Can be changed in the code, but this is the fastest that the Arduino Serial Monitor uses, and its conviniant to use the built in as it automaticly disconnects for you on upload. 
