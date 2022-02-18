# Smart Driving Monitoring

## Fatures Suported:
1. Head Pose
2. Iris Landmarks
3. Drowsiness
4. Object detection such as Mobile Phone, Wine glass, cup, bottle etc.
5. Yawning


## Guidelines

* Open the terminal.

* Clone the repository.
```
git clone https://github.com/chandan5362/Smart-Driver-Monitoring-System.git
```
* Install the required dependencies inside a virtual environment.
```
pip install -r requirements.txt
```
* Run the python script.
```
python driver_analytics.py
```

By default, it accepts input from webcam. Feel free to change the input method and direct the script to the local path of the video. It's not a very fancy project but it can get you started for the advanced version. You can modify it for your purpose. You can also create a PR if you need infomation about the methodology and the algorithms or you want some more features to be included. I have tried my best to write the code as simple as possible. So, Even if you are the novice in python, you can easily understand the the codes.


## Frameworks used
* [Mediapipe](https://google.github.io/mediapipe/) for face detection
* [dlib](https://github.com/davisking/dlib) for iris landmarks
* [YOLOv3](https://pjreddie.com/darknet/yolo/) for object detection.
* [WheNet](https://arxiv.org/abs/2005.10353) for headpose estimation.






