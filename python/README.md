# Python: Federated Learning Simulation

## Overview
This folder contains the Python simulation of federated learning using NASA bearing dataset.

## Files

### `spec_nasa.py`
Converts raw bearing vibration signals into Mel-spectrograms.
- Input: NASA IMS bearing dataset (text files with shape 20480×8)
- Output: Visual spectrogram plot
- Usage: `python spec_nasa.py`

### `build_dataset.py`
Creates training dataset from multiple bearing files.
- Input: Raw vibration files from 1st_test folder
- Output: `X_mel.npy` (spectrograms), `y_labels.npy` (0=healthy, 1=faulty)
- Usage: `python build_dataset.py`

### `fl_simulation.py`
Simulates federated learning with 2 virtual clients (models A and B).
- Model A learns on healthy bearing data
- Model B learns on faulty bearing data
- Models exchange weights and improve together
- Usage: `python fl_simulation.py`

## Quick Start

Install dependencies
pip install -r requirements.txt

Step 1: Build dataset
python build_dataset.py

Step 2: Run FL simulation
python fl_simulation.py

## Expected Output
=== Federated Learning Simulation ===

--- Round 1 ---
Model A - Before: 0.500, After: 0.667
Model B - Before: 0.333, After: 0.667
Accuracy should increase with each round.
