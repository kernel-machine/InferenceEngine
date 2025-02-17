from flask import Flask, Response
import cv2
from pathlib import Path
import zmq
import struct
import numpy as np
import time

app = Flask(__name__)

context = zmq.Context()
frame_generator = context.socket(zmq.REQ)
frame_generator.connect ("tcp://127.0.0.1:5002")

def decode_images(data) -> tuple[list[cv2.Mat], bool]:
    """Estrae le immagini dal buffer binario ricevuto."""
    images = []
    offset = 0
    while offset < len(data):
        # Legge la dimensione dell'immagine
        img_size, is_infested = struct.unpack_from("I?", data, offset)
        offset += 5  # Sposta l'offset
        # Legge l'immagine
        img_bytes = struct.unpack_from(f"{img_size}s", data, offset)[0]
        offset += img_size  # Sposta l'offset
        # Decodifica e aggiunge alla lista
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        images.append(image)
    return images, is_infested

def generate_frames():
    while True:
        frame_generator.send(b"frames")
        message = frame_generator.recv()
        images, label = decode_images(message)
        print(f"image shape: {images[0].shape}")
            # Encode frame as JPEG
        for frame in images:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>Webcam Streaming</h1><img src='/video_feed' width='1280' height='720'/>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)
