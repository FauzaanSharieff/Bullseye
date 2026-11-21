"""Bullseye: Hailo YOLO detection + P-controlled Arduino pan/tilt tracking.

Running on Raspberry Pi 5: python tracking.py

"""

import math
import time

import cv2
import numpy as np
import serial
from picamera2 import Picamera2
from picamera2.devices import Hailo


# ============================================================================
# CONFIGURATION -- START HERE
# Ranges below are suggested *tuning ranges*, not universal physical limits.
# ============================================================================

# Detection / camera (preserved from detection.py)
MODEL = "/usr/share/hailo-models/yolov8s_h8.hef"
TARGET_CLASS = 76                 # COCO class ID: 0-79 (76 = scissors)
TARGET_NAME = "scissors"
CONFIDENCE = 0.5                 # Typical: 0.30-0.80 (0.0-1.0 mathematically)
CAMERA_WIDTH = 1280              # Pixels: match the Picamera2 capture configuration
CAMERA_HEIGHT = 720              # Pixels: match the Picamera2 capture configuration

# Serial / initial servo position (same protocol as pantilt.py)
PORT = "/dev/ttyUSB0"             # Change to the Arduino's current serial port
BAUD_RATE = 115200                # Must match the Arduino sketch
INITIAL_PAN = 90                  # Degrees: within SERVO_MIN..SERVO_MAX
INITIAL_TILT = 90                 # Degrees: within SERVO_MIN..SERVO_MAX
SERVO_MIN = 10                    # Degrees: requested mechanical lower bound
SERVO_MAX = 170                   # Degrees: requested mechanical upper bound

# Control direction: only +1 or -1. Flip if that axis moves the wrong way.
PAN_DIRECTION = -1                # Positive X pixel error -> decrease pan angle
TILT_DIRECTION = +1               # Positive Y pixel error -> increase tilt angle

# P controller: speed (deg/s) = direction * K_P * pixel_error_outside_deadband
K_P_PAN = 0.12                    # Suggested: 0.02-0.40 deg/(s*pixel)
K_P_TILT = 0.12                   # Suggested: 0.02-0.40 deg/(s*pixel)
DEADBAND_X = 15                   # Suggested: 5-40 pixels; 0 disables it
DEADBAND_Y = 15                   # Suggested: 5-40 pixels; 0 disables it

# Motion smoothing / safety
MAX_SPEED = 60.0                  # Suggested: 15-100 deg/s, if servos allow
MAX_ACCEL = 250.0                 # Suggested: 80-600 deg/s^2 (larger = snappier)
LOSS_DECEL = 400.0                # Suggested: 150-800 deg/s^2 on target loss
ERROR_ALPHA = 0.45                # Suggested: 0.20-1.00; 1 = no error smoothing
MAX_STEP = 4.0                    # Suggested: 1-6 degrees per detection update
MAX_DT = 0.20                     # Suggested: 0.05-0.30 s; caps delayed-frame jumps
SERIAL_INTERVAL = 0.04            # Suggested: 0.03-0.10 s; min time between writes

# Intermittent detections / avoiding target switching
LOST_TIMEOUT = 0.60               # Suggested: 0.30-1.50 s before seeking a new target
MATCH_RADIUS = 200.0              # Suggested: 80-350 pixels from last target center
ACQUIRE_FRAMES = 2                # Suggested: 1-4 successive detections to relock

# Preview requires OpenCV GUI support. False avoids the previous cv2.imshow error.
SHOW_PREVIEW = True              # True/False; enable only with GUI-capable OpenCV
LOG_INTERVAL = 0.5                # Suggested: 0.2-2.0 s between terminal status lines


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def approach(value, target, max_change):
    """Move value toward target by no more than max_change."""
    return value + clamp(target - value, -max_change, max_change)


def deadband(error, width):
    """Deadband with no discontinuous jump at its edge."""
    return math.copysign(max(0.0, abs(error) - width), error)


def letterbox(frame, model_width, model_height):
    """Preserve field of view while resizing for the Hailo model."""
    height, width = frame.shape[:2]
    scale = min(model_width / width, model_height / height)
    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(frame, (new_width, new_height))
    pad_left = (model_width - new_width) // 2
    pad_top = (model_height - new_height) // 2
    pad_right = model_width - new_width - pad_left
    pad_bottom = model_height - new_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return np.ascontiguousarray(padded), scale, pad_left, pad_top


def extract_targets(results, model_width, model_height, scale, pad_x, pad_y):
    """Return valid target-class boxes in original CAMERA pixel coordinates."""
    targets = []
    for detection in results[TARGET_CLASS]:
        score = float(detection[4])
        if score < CONFIDENCE:
            continue

        # Hailo output order: normalized (y0, x0, y1, x1).
        y0, x0, y1, x1 = detection[:4]
        x0 = int((float(x0) * model_width - pad_x) / scale)
        x1 = int((float(x1) * model_width - pad_x) / scale)
        y0 = int((float(y0) * model_height - pad_y) / scale)
        y1 = int((float(y1) * model_height - pad_y) / scale)

        x0 = int(clamp(x0, 0, CAMERA_WIDTH - 1))
        x1 = int(clamp(x1, 0, CAMERA_WIDTH - 1))
        y0 = int(clamp(y0, 0, CAMERA_HEIGHT - 1))
        y1 = int(clamp(y1, 0, CAMERA_HEIGHT - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        targets.append({
            "box": (x0, y0, x1, y1),
            "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            "score": score,
        })
    return targets


class TargetSelector:
    """Keep following a nearby detection; relock after an extended loss."""

    def __init__(self):
        self.last_center = None
        self.last_seen = None
        self.pending_center = None
        self.pending_count = 0

    def select(self, detections, now):
        if self.last_center is not None and now - self.last_seen <= LOST_TIMEOUT:
            nearby = [
                item for item in detections
                if math.dist(item["center"], self.last_center) <= MATCH_RADIUS
            ]
            if nearby:
                chosen = min(
                    nearby,
                    key=lambda item: math.dist(item["center"], self.last_center),
                )
                self.last_center = chosen["center"]
                self.last_seen = now
                return chosen, "TRACKING"
            return None, "TEMPORARILY LOST"

        # No recent locked target: confirm a fresh candidate before moving.
        self.last_center = None
        self.last_seen = None
        if not detections:
            self.pending_center = None
            self.pending_count = 0
            return None, "SEARCHING"

        chosen = max(detections, key=lambda item: item["score"])
        if (
            self.pending_center is not None
            and math.dist(chosen["center"], self.pending_center) <= MATCH_RADIUS
        ):
            self.pending_count += 1
        else:
            self.pending_count = 1

        self.pending_center = chosen["center"]
        if self.pending_count >= ACQUIRE_FRAMES:
            self.last_center = chosen["center"]
            self.last_seen = now
            self.pending_center = None
            self.pending_count = 0
            return chosen, "TRACKING"
        return None, "ACQUIRING"


class PanTiltController:
    """P-on-pixel-error velocity control with servo speed/acceleration limits."""

    def __init__(self):
        self.pan = float(clamp(INITIAL_PAN, SERVO_MIN, SERVO_MAX))
        self.tilt = float(clamp(INITIAL_TILT, SERVO_MIN, SERVO_MAX))
        self.v_pan = 0.0
        self.v_tilt = 0.0
        self.filtered_x = None
        self.filtered_y = None

    def update(self, target_center, dt):
        dt = clamp(dt, 0.0, MAX_DT)
        if target_center is None:
            desired_v_pan = desired_v_tilt = 0.0
            accel = LOSS_DECEL  # Stop gently on momentary detection loss.
            self.filtered_x = None
            self.filtered_y = None
        else:
            raw_x = target_center[0] - CAMERA_WIDTH / 2.0
            raw_y = target_center[1] - CAMERA_HEIGHT / 2.0

            # Exponential moving average to reduce bounding-box jitter.
            self.filtered_x = (
                raw_x if self.filtered_x is None else
                ERROR_ALPHA * raw_x + (1.0 - ERROR_ALPHA) * self.filtered_x
            )
            self.filtered_y = (
                raw_y if self.filtered_y is None else
                ERROR_ALPHA * raw_y + (1.0 - ERROR_ALPHA) * self.filtered_y
            )

            error_x = deadband(self.filtered_x, DEADBAND_X)
            error_y = deadband(self.filtered_y, DEADBAND_Y)

            desired_v_pan = clamp(
                PAN_DIRECTION * K_P_PAN * error_x, -MAX_SPEED, MAX_SPEED,
            )
            desired_v_tilt = clamp(
                TILT_DIRECTION * K_P_TILT * error_y, -MAX_SPEED, MAX_SPEED,
            )
            accel = MAX_ACCEL

        # Acceleration limiting: command speed cannot change instantly.
        self.v_pan = approach(self.v_pan, desired_v_pan, accel * dt)
        self.v_tilt = approach(self.v_tilt, desired_v_tilt, accel * dt)

        # Limit the position change per received camera frame as well.
        pan_step = clamp(self.v_pan * dt, -MAX_STEP, MAX_STEP)
        tilt_step = clamp(self.v_tilt * dt, -MAX_STEP, MAX_STEP)
        self.pan = clamp(self.pan + pan_step, SERVO_MIN, SERVO_MAX)
        self.tilt = clamp(self.tilt + tilt_step, SERVO_MIN, SERVO_MAX)

        # Do not continue pushing into a physical limit.
        if (self.pan <= SERVO_MIN and self.v_pan < 0) or (
            self.pan >= SERVO_MAX and self.v_pan > 0
        ):
            self.v_pan = 0.0
        if (self.tilt <= SERVO_MIN and self.v_tilt < 0) or (
            self.tilt >= SERVO_MAX and self.v_tilt > 0
        ):
            self.v_tilt = 0.0

        return round(self.pan), round(self.tilt)


def send_angles(arduino, pan, tilt):
    """Exactly the 'pan,tilt\\n' format from pantilt.py."""
    pan = int(clamp(pan, SERVO_MIN, SERVO_MAX))
    tilt = int(clamp(tilt, SERVO_MIN, SERVO_MAX))
    arduino.write(f"{pan},{tilt}\n".encode("utf-8"))


def preview_supported():
    """Avoid calling imshow when OpenCV has no GUI backend."""
    if not SHOW_PREVIEW:
        return False
    try:
        backend = cv2.currentUIFramework()
    except (AttributeError, cv2.error):
        backend = None
    if not backend:
        print("OpenCV has no GUI backend. Continuing without a preview window.")
        return False
    return True


def main():
    if SERVO_MIN >= SERVO_MAX:
        raise ValueError("SERVO_MIN must be less than SERVO_MAX")
    if PAN_DIRECTION not in (-1, 1) or TILT_DIRECTION not in (-1, 1):
        raise ValueError("Direction values must be either +1 or -1")
    if not (0 < ERROR_ALPHA <= 1):
        raise ValueError("ERROR_ALPHA must be in (0, 1]")
    if ACQUIRE_FRAMES < 1:
        raise ValueError("ACQUIRE_FRAMES must be at least 1")

    arduino = None
    camera = None
    show_preview = preview_supported()
    controller = PanTiltController()
    selector = TargetSelector()

    try:
        # Open serial BEFORE starting Hailo; Uno typically reboots on connection.
        arduino = serial.Serial(PORT, BAUD_RATE, timeout=1, write_timeout=1)
        time.sleep(2)
        send_angles(arduino, round(controller.pan), round(controller.tilt))
        last_sent = (round(controller.pan), round(controller.tilt))
        last_send_time = time.monotonic()
        print(f"Arduino connected on {PORT}, initial pan/tilt: {last_sent}")

        with Hailo(MODEL) as hailo:
            model_height, model_width, _ = hailo.get_input_shape()
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "RGB888",
                }
            )
            camera.configure(config)
            camera.start()
            print(f"Tracking {TARGET_NAME} (COCO ID {TARGET_CLASS})")
            print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}; model: {model_width}x{model_height}")
            print("Press Ctrl+C to stop" + ("; Q or Esc also works" if show_preview else ""))

            previous_time = time.monotonic()
            last_log_time = previous_time

            while True:
                frame = camera.capture_array()
                inference_frame, scale, pad_x, pad_y = letterbox(
                    frame, model_width, model_height,
                )
                results = hailo.run(inference_frame)
                now = time.monotonic()
                dt = now - previous_time
                previous_time = now

                detections = extract_targets(
                    results, model_width, model_height, scale, pad_x, pad_y,
                )
                selected, status = selector.select(detections, now)
                center = selected["center"] if selected is not None else None

                pan, tilt = controller.update(center, dt)
                angles = (pan, tilt)
                if angles != last_sent and now - last_send_time >= SERIAL_INTERVAL:
                    send_angles(arduino, pan, tilt)
                    last_sent = angles
                    last_send_time = now

                if now - last_log_time >= LOG_INTERVAL:
                    if selected is None:
                        detail = "no confirmed target"
                    else:
                        cx, cy = center
                        detail = (
                            f"center=({cx:.0f}, {cy:.0f}) "
                            f"error=({cx-CAMERA_WIDTH/2:+.0f}, "
                            f"{cy-CAMERA_HEIGHT/2:+.0f}) px "
                            f"confidence={selected['score']:.2f}"
                        )
                    print(
                        f"{status:18s} | {detail} | "
                        f"pan={pan} tilt={tilt}", flush=True,
                    )
                    last_log_time = now

                if show_preview:
                    # Original camera coordinate system (after undoing letterbox).
                    cv2.drawMarker(
                        frame, (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2),
                        (255, 0, 0), cv2.MARKER_CROSS, 25, 2,
                    )
                    if selected is not None:
                        x0, y0, x1, y1 = selected["box"]
                        cx, cy = selected["center"]
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.circle(frame, (round(cx), round(cy)), 6, (0, 0, 255), -1)
                        cv2.putText(
                            frame, f"{TARGET_NAME}: {selected['score']:.2f}",
                            (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2,
                        )
                    cv2.putText(
                        frame, f"{status} | Pan {pan} Tilt {tilt}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2,
                    )
                    try:
                        cv2.imshow("Bullseye Tracking", frame)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as err:
                        print(f"Preview unavailable ({err}). Tracking continues.")
                        show_preview = False
                        key = -1
                    if key in (ord("q"), ord("Q"), 27):
                        break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # Servos hold their last sent angle; do not abruptly re-center at exit.
        if camera is not None:
            try:
                camera.stop()
            finally:
                camera.close()
        if arduino is not None:
            arduino.close()
        if show_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("Camera and serial connection closed.")


if __name__ == "__main__":
    main()
