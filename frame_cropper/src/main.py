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
import asyncio
import queue
import threading

context = zmq.Context()
frame_generator = context.socket(zmq.REQ)
frame_generator.connect ("tcp://frame-generator:5001")

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
MIN_BUFFER_SIZE = 3
MAX_BUFFER_SIZE = 6
buffer = queue.Queue(maxsize=MAX_BUFFER_SIZE)

def add_frames():
    while True:
        if buffer.qsize() < MAX_BUFFER_SIZE:
            frame_generator.send(b"frames")
            message = frame_generator.recv()
            images, _ = decode_images(message)
            def frame_proc(x):
                x = vs.crop_frame(x)
                return cv2.resize(x,(1280,720))
            images = list(map(frame_proc, images))
            # Shared resource
            buffer.put(images)
        else:
            time.sleep(0.1)

def get_frames():
    while True:
        message = frame_requester.recv()
        if message==b"frames":
            images = buffer.get()
            images_bytes = []
            for img in images:
                _, encoded = cv2.imencode(".jpg", img)
                img_bytes = encoded.tobytes()
                img_size = len(img_bytes)
                images_bytes.append(struct.pack(f"I?{img_size}s", img_size, True, img_bytes))
            frame_requester.send(b"".join(images_bytes))

    
add_thread = threading.Thread(target=add_frames)
remove_thread = threading.Thread(target=get_frames)
add_thread.start()
remove_thread.start()

add_thread.join()
remove_thread.join()