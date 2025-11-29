# Bill of Materials (BOM) — Hardware Components

## Stage 2: Sensors & Microcontroller

| Component | Model | Quantity | Price (RUB) | Link/Notes |
|-----------|-------|----------|-------------|-----------|
| Microcontroller | ESP32-S3 DevKit | 2 | 500-700 | Has AI-Vector instructions |
| Microphone | INMP441 (I2S) | 2 | 200-300 | Digital I2S, 20 kHz sampling |
| OLED Display | 0.96" SSD1306 | 1 | 150-200 | Optional, for status display |
| CPU Fan/Motor | 12V DC brushless | 2 | 50-100 | For vibration testing |
| Tape | Insulating tape | 1 | 20 | To induce vibration fault |
| Wires | DuPont, 22AWG | 1 pack | 50-100 | For connections |
| USB Cable | Micro-USB | 2 | 30-50 | For programming ESP32 |
| **Total Cost** | | | **1000-1500** | *Approximate* |

## Connection Diagram

### ESP32-S3 ↔ INMP441 Microphone

3.3V VDD Power
GND GND Ground
GPIO12 SCK I2S Serial Clock
GPIO13 WS I2S Word Select (LR Clock)
GPIO14 SD I2S Serial Data
ESP32-S3 ↔ Motor Setup
12V Power Supply
↓
Motor A (Healthy) — Microphone A — ESP32-S3 #1
Motor B (Faulty*) — Microphone B — ESP32-S3 #2
↓
WiFi Router (for MQTT communication)


*Fault simulation: Attach small tape piece to motor blade to create vibration

## Assembly Notes

1. Use breadboard or custom PCB for connections
2. Add 10µF capacitor between VDD and GND on microphone module
3. Use short wires (<10cm) to minimize noise
4. Mount microphones close to motor bearings
5. Keep motors away from WiFi interference sources
