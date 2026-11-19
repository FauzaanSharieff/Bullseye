
import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo

# ==========================================
# CONFIGURATION
# ==========================================

MODEL = "/usr/share/hailo-models/yolov8s_h8.hef"
CONFIDENCE = 0.5

TARGET_CLASS = 76
TARGET_NAME = "scissors"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ==========================================
# COCO OBJECT CLASSES (ID: NAME)
# ==========================================
#  0: person           1: bicycle          2: car
#  3: motorcycle       4: airplane         5: bus
#  6: train            7: truck            8: boat
#  9: traffic light   10: fire hydrant    11: stop sign
# 12: parking meter   13: bench           14: bird
# 15: cat             16: dog             17: horse
# 18: sheep           19: cow             20: elephant
# 21: bear            22: zebra           23: giraffe
# 24: backpack        25: umbrella        26: handbag
# 27: tie             28: suitcase        29: frisbee
# 30: skis            31: snowboard       32: sports ball
# 33: kite            34: baseball bat    35: baseball glove
# 36: skateboard      37: surfboard       38: tennis racket
# 39: bottle          40: wine glass      41: cup
# 42: fork            43: knife           44: spoon
# 45: bowl            46: banana          47: apple
# 48: sandwich        49: orange          50: broccoli
# 51: carrot          52: hot dog         53: pizza
# 54: donut           55: cake            56: chair
# 57: couch           58: potted plant    59: bed
# 60: dining table    61: toilet          62: tv
# 63: laptop          64: mouse           65: remote
# 66: keyboard        67: cell phone      68: microwave
# 69: oven            70: toaster         71: sink
# 72: refrigerator    73: book            74: clock
# 75: vase            76: scissors        77: teddy bear
# 78: hair drier      79: toothbrush


# ==========================================
# LETTERBOX PREPROCESSING
# ==========================================

def letterbox(frame, model_width, model_height):
    height, width = frame.shape[:2]

    scale = min(model_width / width,
                model_height / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(
        frame, (new_width, new_height)
    )

    pad_left = (model_width - new_width) // 2
    pad_top = (model_height - new_height) // 2

    pad_right = model_width - new_width - pad_left
    pad_bottom = model_height - new_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return np.ascontiguousarray(padded), scale, pad_left, pad_top


# ==========================================
# INITIALIZE CAMERA AND HAILO
# ==========================================

with Hailo(MODEL) as hailo:

    model_height, model_width, _ = hailo.get_input_shape()

    camera = Picamera2()

    config = camera.create_preview_configuration(
        main={
            "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            "format": "RGB888"
        }
    )

    camera.configure(config)
    camera.start()

    print(f"Tracking: {TARGET_NAME}")
    print(f"Camera: {CAMERA_WIDTH} x {CAMERA_HEIGHT}")
    print(f"YOLO input: {model_width} x {model_height}")
    print("Press Q or ESC to quit")

    try:
        while True:

            # Capture wide camera frame
            frame = camera.capture_array()

            # Resize and pad without cropping
            inference_frame, scale, pad_x, pad_y = letterbox(
                frame, model_width, model_height
            )

            # Run YOLO on Hailo-8
            results = hailo.run(inference_frame)

            # Find highest-confidence target
            best_detection = None
            best_score = CONFIDENCE

            for detection in results[TARGET_CLASS]:
                score = float(detection[4])

                if score > best_score:
                    best_score = score
                    best_detection = detection

            if best_detection is not None:

                # Normalized coordinates from YOLO
                y0, x0, y1, x1 = best_detection[:4]

                # Convert to model pixel coordinates
                x0 = x0 * model_width
                x1 = x1 * model_width
                y0 = y0 * model_height
                y1 = y1 * model_height

                # Undo letterbox padding and scaling
                x0 = int((x0 - pad_x) / scale)
                x1 = int((x1 - pad_x) / scale)
                y0 = int((y0 - pad_y) / scale)
                y1 = int((y1 - pad_y) / scale)

                # Keep coordinates within camera image
                x0 = max(0, min(x0, CAMERA_WIDTH - 1))
                x1 = max(0, min(x1, CAMERA_WIDTH - 1))
                y0 = max(0, min(y0, CAMERA_HEIGHT - 1))
                y1 = max(0, min(y1, CAMERA_HEIGHT - 1))

                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2

                # Draw bounding box
                cv2.rectangle(
                    frame, (x0, y0), (x1, y1),
                    (0, 255, 0), 2
                )

                # Draw label and confidence
                cv2.putText(
                    frame,
                    f"{TARGET_NAME}: {best_score:.2f}",
                    (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

                # Draw target center
                cv2.circle(
                    frame, (center_x, center_y),
                    6, (0, 0, 255), -1
                )

                # Print coordinates
                print(
                    f"{TARGET_NAME} | "
                    f"Center: ({center_x}, {center_y}) px | "
                    f"Confidence: {best_score:.2f}",
                    flush=True
                )

            # Draw camera center
            cv2.drawMarker(
                frame,
                (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                25, 2
            )

            cv2.imshow("Bullseye Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
