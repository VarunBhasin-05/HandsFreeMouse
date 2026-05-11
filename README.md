# AI-Powered Virtual Mouse

This project implements a gesture-based virtual mouse using Python, OpenCV, and MediaPipe. It allows users to control their system cursor and perform clicks using hand movements captured via a standard webcam.

## Technical Highlights
* **Performance**: Replaced standard automation libraries with `pynput` for lower latency and real-time responsiveness.
* **Precision**: Implemented an exponential smoothing algorithm to eliminate cursor jitter.
* **Coordinate Mapping**: Used linear interpolation to map webcam coordinates to the user's specific screen resolution.
* **Active Boundary**: Designed a centered "detection box" to allow full-screen navigation with minimal physical hand movement.

## Features
* **Navigation**: Move the cursor using the index finger.
* **Clicking**: Perform a left-click by "pinching" the index and middle fingers.
* **Visual Debugging**: On-screen landmarks and status indicators for real-time feedback.

## Installation & Usage
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the script:
   `python main.py`
3. Use the Index finger to move. Pinch index and middle fingers to click. Press `ESC` to exit.
