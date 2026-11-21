# 🎯 Bullseye

**Real-Time Object Detection and Autonomous Camera Tracking**

Bullseye is a computer vision and robotics project designed to detect and autonomously track objects using a two-axis pan-tilt camera system.

The system uses a Raspberry Pi 5 with a Hailo-8 AI accelerator to run YOLO-based object detection in real time. An Arduino Uno controls the pan-tilt servos, allowing the camera to follow a selected target.

## System Architecture

```text
      Raspberry Pi Camera Module 3
                    |
                    v
              Raspberry Pi 5
                    |
                    v
        YOLOv8 Object Detection
             (Hailo-8 NPU)
                    |
                    v
           Target Coordinates
                    |
                    v
          Pan-Tilt P-Controller
                    |
                    v
          Serial Communication
                    |
                    v
               Arduino Uno
                    |
                    v
           Pan and Tilt Servos
```

## Hardware

| Component | Description |
|---|---|
| Raspberry Pi 5 | Main processing unit |
| Raspberry Pi Camera Module 3 | Image acquisition |
| Raspberry Pi AI HAT+ (26 TOPS) | Hailo-8 neural network accelerator |
| Arduino Uno | Servo control and serial communication |
| Two servo motors | Pan and tilt actuation |
| Pan-tilt mechanism | Camera mounting and movement |

## Software & Dependencies

| Technology | Purpose |
|---|---|
| Python | Main application |
| Picamera2 | Camera interface |
| YOLOv8 Small | Object detection |
| HailoRT | Hardware-accelerated inference |
| OpenCV | Image processing and visualization |
| NumPy | Numerical and image operations |
| PySerial | Raspberry Pi–Arduino communication |
| Arduino C++ | Low-level motor control |

The project uses a pretrained YOLOv8 model compiled for Hailo-8 (`yolov8s_h8.hef`), allowing inference directly on the AI accelerator without requiring the Ultralytics library at runtime.


## Note

Bullseye was developed as a sub-unit of a much larger project. Also, it was just a project that was supposed to help me get familiar with microcomputers and their peripherals. Therefore, a large part of the project was vibecoded because figuring out the P-controller parameters or reading Hailo documentation to use its Picamera2 implementation myself was not something I was willing to spend my time on. Nevertheless, I learnt a lot from this project once it all came together..
