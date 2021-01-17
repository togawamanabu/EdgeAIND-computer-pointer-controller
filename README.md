# Computer Pointer Controller

This is eye gaze mouse controller application with OpenVino.

Used 4 OpenVino models

 - Face Detection
     Detect Face position in the frame 
 - Head Pose Estimation
     Estimate Head pose angles for gaze estimation
 - Facial Landmarks Detection
     Left and Right eye detection for gaze estimation
 - Gaze Estimation Model
     Gaze estimate from head pose and eye images

## Project Set Up and Installation

### Download models

    cd /opt/intel/openvino_2021/deployment_tools/tools/model_downloader
     ./downloader.py --name face-detection-adas-0001 -o /home/ubuntu/udacity/edigeAi/EdgeAIND-computer-pointer-controller/models
     ./downloader.py --name head-pose-estimation-adas-0001 -o /home/ubuntu/udacity/edigeAi/EdgeAIND-computer-pointer-controller/models
     ./downloader.py --name gaze-estimation-adas-0002 -o /home/ubuntu/udacity/edigeAi/EdgeAIND-computer-pointer-controller/models

### Install pyautogui
     python3 -m pip install pyautogui
     
     sudo apt-get install scrot

     sudo apt-get install python3-tk

     sudo apt-get install python3-dev


## Demo
    #with video file 
    python3 main.py -i ../bin/demo.mp4 -s True

    #with camera
    python3 main.py -i cam -s True 

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
