# Federated Learning System for Bearing Fault Diagnosis

**Project for Tsinghua University Admission**

## Quick Overview
- **Goal**: Build a distributed ML system that learns bearing faults without sharing raw data
- **Hardware**: ESP32-S3 microcontrollers + INMP441 microphones
- **Method**: Federated Learning (each device trains locally, shares only weights)
- **Status**: Stage 1 (Python simulation) ✅ | Stage 2-6 (In Progress)

## Directory Structure

motor_fedlearning/
├── python/ # FL Simulation (Python) — STAGE 1 ✅
│ ├── spec_nasa.py # Vibration → Spectrogram
│ ├── build_dataset.py # Create training data
│ ├── fl_simulation.py # FL with 2 models
│ ├── requirements.txt # Python dependencies
│ └── README.md # Python docs
├── hardware/ # Components & Circuits — STAGE 2
│ ├── BOM.md # Parts list
│ └── circuit_diagram.txt
├── firmware/ # ESP32 C++ Code — STAGE 4 (TBD)
├── server/ # MQTT Aggregator — STAGE 5 (TBD)
├── data/ # NASA Bearing Dataset
├── .gitignore # Git exclusions
└── README.md # This file


## Quick Start

### 1. Setup Python Environment
python -m venv venv
venv\Scripts\activate
pip install -r python/requirements.txt

### 2. Build Dataset
cd python
python build_dataset.py

### 3. Run FL Simulation
python fl_simulation.py

## Project Stages

| Stage | Duration | Status | Output |
|-------|----------|--------|--------|
| 1. Python FL Simulation | 2 weeks | ✅ DONE | `fl_simulation.py` working |
| 2. Hardware Setup | 1 week | 📦 TBD | Components ordered |
| 3. Edge Impulse Training | 3 weeks | 📦 TBD | C++ library exported |
| 4. On-Device Training | 4 weeks | 📦 TBD | Firmware with SGD |
| 5. IoT Communication | 2 weeks | 📦 TBD | MQTT aggregation |
| 6. Patent & Demo | 2 weeks | 📦 TBD | Video + Роспатент filing |

## Key Technologies
- **ML Framework**: TensorFlow 2.x + TensorFlow Federated
- **Edge Device**: ESP32-S3 (ARM Cortex-M7, 240 MHz)
- **Audio Processing**: librosa (Mel-spectrograms)
- **Communication**: MQTT (for weight sync)
- **IoT Platform**: Edge Impulse (for inference)

## Patent Information
**Title**: Система вибродиагностики с распределённым обучением  
**Status**: Preparing for submission to Роспатент (Russian IP Office)

## Contact
Project for Tsinghua University Scholarship Application

---

**Last Updated**: Nov 2025

