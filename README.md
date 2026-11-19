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
          Pan-Tilt Controller
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

*The tracking controller and automated servo integration are currently under development.*

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

## Current Status

**Object detection pipeline operational.**

The system currently supports:

- Real-time, Hailo-accelerated object detection
- Selection of a target from the 80 COCO object classes
- Bounding-box visualization and target coordinate extraction
- Raspberry Pi–Arduino serial communication

**Next milestone:** Integrate the detection pipeline with pan-tilt motor control to achieve autonomous object tracking.

## Getting Started

### System dependencies

On Raspberry Pi OS:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-serial hailo-all
```

### Python environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### Run object detection

```bash
python detection.py
```

The target object can be configured in `detection.py` using its COCO class ID.

## Project Status

🚧 **Under active development**

The object detection stage is complete. Autonomous tracking and closed-loop pan-tilt control are planned for the next stage.