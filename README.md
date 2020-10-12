# Complex YOLO ROS

### More details
* https://github.com/ghimiredhikura/Complex-YOLOv3

### Environment
* Jetson AGX Xavier 
* CUDA 10.2
* CUDNN 8.0
* PyTorch 1.6.0
* torchvision 0.6.0
* ROS melodic

### Installation

```
$ cd <CATKIN_WORKSPACE>/src
$ git clone https://github.com/wonseok-Jang/complex_yolo_ros
$ cd <CATKIN_WORKSPACE> && catkin_make -j <CPU cores>
$ roslaunch complex_yolo_ros complex_yolov3.launch
```

### Set param (In `launch/complex_yolov3.launch`)
* `model_def`: Configure file path
* `weights_path`: Weights path
* `class_path`: Name path
* `nms_thres`: Non-Maximum-Suppression threshold
* `conf_thres`: Confidence threshold

### Subscribed topics
* `/velodyne_points` (You can set topic name in `launch/complex_yolov3.launch`)

### Published topics
* `/complex_yolo_ros/detected_objects`: complex_yolo_ros::DetectedObjectArray
