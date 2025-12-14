#include <I2S.h>

const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 4096;

int16_t audio_buffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== ESP32-S3 I2S Audio Test ===");

  I2S.setRxMode();
  if (!I2S.begin(I2S_PHILIPS_MODE, SAMPLE_RATE, 16)) {
    Serial.println("FAILED to initialize I2S!");
    while(1);
  }
  
  Serial.println("I2S initialized successfully!");
  Serial.println("Recording audio...");
}

void loop() {
  int bytes_read = I2S.readBytes(
    (char*)audio_buffer, 
    BUFFER_SIZE * 2  
  );
  
  if (bytes_read > 0) {
    int samples = bytes_read / 2;
    Serial.print("Read ");
    Serial.print(samples);
    Serial.println(" samples");
    
    Serial.print("First 10 samples: ");
    for (int i = 0; i < 10 && i < samples; i++) {
      Serial.print(audio_buffer[i]);
      Serial.print(" ");
    }
    Serial.println();
  }
  
  delay(100);
}
