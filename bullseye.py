import serial
import time

# Serial configuration
PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# Connect to Arduino
arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)

# Allow Arduino to restart after connection
time.sleep(2)

print("Connected to Arduino Uno!")
print("Enter angles as: pan,tilt")
print("Type 'q' to quit.\n")

try:
    while True:
        user_input = input("Enter angles: ").strip()

        if user_input.lower() == "q":
            break

        try:
            pan, tilt = map(int, user_input.split(","))

            # Validate angles
            if not (0 <= pan <= 180 and 0 <= tilt <= 180):
                print("Angles must be between 0 and 180.")
                continue

            # Send angles to Arduino
            command = f"{pan},{tilt}\n"
            arduino.write(command.encode("utf-8"))

            print(f"Sent: Pan={pan}°, Tilt={tilt}°")

        except ValueError:
            print("Invalid input! Use format: 90,120")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    arduino.close()
    print("Serial connection closed.")
