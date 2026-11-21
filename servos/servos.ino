
 /*
void setup()
{
}
void loop()
{
}
 */

// /*

#include <Servo.h>

Servo pan;
Servo tilt;

char buffer[32];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);

  pan.attach(9);
  tilt.attach(11);

  pan.write(90);
  tilt.write(90);
  delay(1000);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      buffer[bufferIndex] = '\0';

      int panAngle, tiltAngle;
      char extra;

      // Accept exactly two integers
      if (sscanf(buffer, " %d , %d %c",
                 &panAngle, &tiltAngle, &extra) == 2) {

        if (panAngle >= 0 && panAngle <= 180 &&
            tiltAngle >= 0 && tiltAngle <= 180) {

          // Pan servo movement
          pan.write(panAngle);
          delay(20);

          // Tilt servo movement
          tilt.write(tiltAngle);
          delay(20);
        }
      }

      bufferIndex = 0;
    }
    else if (c != '\r') {
      if (bufferIndex < sizeof(buffer) - 1) {
        buffer[bufferIndex++] = c;
      }
      else {
        bufferIndex = 0;
      }
    }
  }
}



// */
