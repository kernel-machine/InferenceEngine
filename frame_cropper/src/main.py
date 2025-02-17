import argparse
import glob
import os
import zmq
import cv2
import struct
import random
import time
import numpy as np
from VideoSegmenter import VideoSegmenter
from frame_buffer import FrameBuffer

context = zmq.Context()
frame_generator = context.socket(zmq.REQ)
frame_generator.connect ("tcp://127.0.0.1:5001")

context2 = zmq.Context()
frame_requester = context2.socket(zmq.REP)
frame_requester.bind("tcp://*:5002")

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

vs = VideoSegmenter()

while True:
    message = frame_requester.recv()
    if message==b"frames":
        frame_generator.send(b"frames")
        message = frame_generator.recv()
        images, label = decode_images(message)

        images_bytes = []
        for img in images:
            img = vs.crop_frame(img)
            img = cv2.resize(img, (1280,720))
            _, encoded = cv2.imencode(".jpg", img)
            img_bytes = encoded.tobytes()
            img_size = len(img_bytes)
            images_bytes.append(struct.pack(f"I?{img_size}s", img_size, label, img_bytes))
            
        frame_requester.send(b"".join(images_bytes))


    
