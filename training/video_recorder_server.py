from flask import Flask, request
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)

RECORDING = False
video_writer = None
SAVE_DIR = "recorded_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 10  # Match the Pi frame rate

@app.route('/start', methods=['POST'])
def start_recording():
    global RECORDING, video_writer

    if RECORDING:
        return "Already recording", 200

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"video_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    RECORDING = True
    print(f"Recording started: {filename}")
    return "Recording started", 200

@app.route('/stop', methods=['POST'])
def stop_recording():
    global RECORDING, video_writer

    if not RECORDING:
        return "Not recording", 200

    RECORDING = False
    video_writer.release()
    video_writer = None
    print("Recording stopped and saved.")
    return "Recording stopped", 200

@app.route('/upload', methods=['POST'])
def upload_frame():
    global RECORDING, video_writer

    if not RECORDING:
        return "Not recording", 403

    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    video_writer.write(frame)
    return "Frame received", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)


#curl -X POST http://<PC-IP>:8000/start
#curl -X POST http://<PC-IP>:8000/stop