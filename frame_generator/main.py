import argparse
import glob
import os
import zmq
import cv2
from frame_buffer import FrameBuffer
import struct
import random

args = argparse.ArgumentParser()
args.add_argument("--video", type=str, required=True, help="Path to the video to process")
args.add_argument("--window_size", type=int, default=32)
args = args.parse_args()

context = zmq.Context()
socket = context.socket(zmq.REP)  # Push mode (one-way)
socket.bind("tcp://*:5001")  # Ascolta su porta 5555

if os.path.isdir(args.video):
    while True:
        video_paths = glob.glob(os.path.join(args.video,"*","*.mkv"))
        random.shuffle(video_paths)
        for video in video_paths:
            class_name = os.path.split(os.path.dirname(video))[1]
            is_infested = "infested" in class_name
            frame_buffer = FrameBuffer(args.window_size)

            video = cv2.VideoCapture(video)
            success = True
            while success:
                success, frame = video.read()
                # Scale frame to 4k keeping aspect ratio
                if success:
                    height , width = frame.shape[:2]
                    frame = cv2.resize(frame, (int(width*0.25), int(height*0.25)))
                    frame_buffer.append(frame)
            print("Sending segments")
            for segment in frame_buffer.get_segments():
                print("Wait for request...")
                message = socket.recv()
                print(f"Received: {message}")
                if message==b"frames":
                    images = []
                    for img in segment:
                        _, encoded = cv2.imencode(".jpg", img)
                        img_bytes = encoded.tobytes()
                        img_size = len(img_bytes)
                        images.append(struct.pack(f"I?{img_size}s", img_size, is_infested, img_bytes))
                    print("Sending frames")
                    socket.send(b"".join(images))

