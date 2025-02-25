from flask import Flask, Response
import cv2
import sys
import signal
import time

app = Flask(__name__)

"""
# Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# For webcams, we use V4L2
video_capture = cv2.VideoCapture(camera_id)
# How to set video capture properties using V4L2:
# Full list of Video Capture Properties for OpenCV: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
#Select Pixel Format:
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
# Two common formats, MJPG and H264
# video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Default libopencv on the Jetson is not linked against libx264, so H.264 is not available
# video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
# Select frame size, FPS:
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)
# camera = cv2.VideoCapture(1, cv2.CAP_V4L)
camera_ready = False
while not camera_ready:
    if not (video_capture.isOpened()):
        print("Could not open video device: retrying")
        video_capture.release()
        video_capture = cv2.VideoCapture(camera_id)
        time.sleep(3)
    else:
        camera_ready = True
"""

"""
def handler(signal, frame):
    print("\nGracefully shutting down...")
    # nonlocal video_capture
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()  # Release the camera resource
        print("Camera released.")
    sys.exit(0)
signal.signal(signal.SIGINT, handler)
"""

def generate_frames():
    camera_id = "/dev/video2"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not (video_capture.isOpened()):
        print("Could not open video device.")
    else:
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            else:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video_capture.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Webcam Streaming</h1><img src='/video_feed' width='640' height='480'/>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
