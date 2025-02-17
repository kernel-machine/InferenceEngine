import argparse
import glob
import os
from VideoSegmenter import VideoSegmenter
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
socket.bind("tcp://*:5555")  # Ascolta su porta 5555

while True:
    video_paths = glob.glob(os.path.join(args.video,"*","*.mkv"))
    random.shuffle(video_paths)
    for video in video_paths:
        class_name = os.path.split(os.path.dirname(video))[1]
        is_infested = "infested" in class_name
        vs = VideoSegmenter(video, output_size=224)
        frame_buffer = FrameBuffer(args.window_size)

        print(f"Extracting frames")
        for frame in vs.get_frames():
            frame_buffer.append(frame)
        print(f"Frame buffer filled with {len(frame_buffer)} of {is_infested}")

        for segment in frame_buffer.get_segments():
            message = socket.recv()
            if message==b"frames":
                images = []
                for img in segment:
                    _, encoded = cv2.imencode(".jpg", img)
                    img_bytes = encoded.tobytes()
                    img_size = len(img_bytes)
                    images.append(struct.pack(f"I?{img_size}s", img_size, is_infested, img_bytes))
                
                socket.send(b"".join(images))

