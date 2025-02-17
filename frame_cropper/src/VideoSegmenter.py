import cv2
import argparse
import os
import numpy as np
from collections.abc import Generator
import time

MIN_AREA_THRESHOLD = 1200
BOX_SIZE = 224

class VideoSegmenter:
    def __init__(self, output_size=224, show_debug:bool=False) -> None:
        self.output_size = output_size
        self.show_debug = show_debug
        self.processed_frame_counter = 0
        self.background = None
        if self.show_debug:
            video_id = os.path.basename(self.video_path).split(" ")[0]
            video_name = f"video_{video_id}_{int(time.time_ns()/1000)}_debug.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vwriter = cv2.VideoWriter(video_name, fourcc, 20, (2*output_size, output_size*4))
            self.vwriter.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

    def crop_frame(self, frame:cv2.Mat) -> cv2.Mat:
        # Remove timestamp band
        # frame = frame[0:-180][:]
        # (frame_h,frame_w) = frame.shape[:2]

        # frame = cv2.resize(frame, (int(frame_w*0.25), int(frame_h*0.25)))
        # (frame_h,frame_w) = frame.shape[:2]

        # Get Saturation and Value mask
        (frame_h,frame_w) = frame.shape[:2]
        frame_blur = cv2.GaussianBlur(frame, (7,7), 10)
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        saturation_channel = cv2.normalize(frame_hsv[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
        _, mask_saturation = cv2.threshold(saturation_channel, 50, 255, cv2.THRESH_BINARY) # For saturation
        value_channel = cv2.normalize(frame_hsv[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
        _, mask_value = cv2.threshold(value_channel, 60, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(mask_saturation, mask_value)
        self.processed_frame_counter+=1
        if self.background is None or self.processed_frame_counter % 100 == 0:
            self.background = mask
            print("Background updated")
        else:
            motion_mask = cv2.absdiff(self.background, mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            inverted_motion_mask = cv2.bitwise_not(motion_mask)
            contours, _ = cv2.findContours(inverted_motion_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            threshold_area = 300

            # Fill small holes to remove noise
            for c in contours:
                if cv2.contourArea(c) < threshold_area:
                    motion_mask = cv2.drawContours(motion_mask, [c], -1, 0, cv2.FILLED)
            mask = cv2.bitwise_and(mask, motion_mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                print(f"Found {len(contours)} contours")
                contours = list(filter(lambda x:cv2.boundingRect(x)[2]>30, contours)) #Remove small holes
                if len(contours) > 0:
                    areas = list(map(lambda x:cv2.contourArea(x), contours))
                    max_index = np.argmax(areas)
                    if areas[max_index] > MIN_AREA_THRESHOLD: #Take the largest hole
                        c = contours[max_index]
                        x, y, w, h = cv2.boundingRect(c)
                        center = (x+(w//2),y+(h//2))
                        top_left = (center[0]-BOX_SIZE//2,center[1]-BOX_SIZE//2)
                        bottom_right = (center[0]+BOX_SIZE//2,center[1]+BOX_SIZE//2)
                        if bottom_right[1]>frame_h:
                            out_size = bottom_right[1]-frame_h
                            top_left = (top_left[0], top_left[1]-out_size)
                        if top_left[1] < 0:
                            bottom_right = (bottom_right[0], bottom_right[1]-top_left[1])
                            top_left = (top_left[0], 0)                        
                        x1, y1 = top_left
                        x2, y2 = bottom_right
                        # x1, y1 = max(x1, 0), max(y1, 0)
                        # x2, y2 = min(x2, frame_w), min(y2, frame_h)
                        cropped_frame = frame[y1:y2, x1:x2]
                        frame = cv2.circle(frame, center, 5, (0,0,255), 2)
                        frame = cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 20)
                        frame = cv2.putText(frame, f"Area: {areas[max_index]}", bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        return frame
        return frame
        '''
        if self.show_debug and self.vwriter is not None:
            mask_saturation = cv2.cvtColor(mask_saturation, cv2.COLOR_GRAY2BGR)
            mask_value = cv2.cvtColor(mask_value, cv2.COLOR_GRAY2BGR)
            motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = np.vstack([mask_saturation, mask_value, motion_mask, mask,frame])
            frame = cv2.resize(frame, (2*self.output_size, self.output_size*4))
            #self.vwriter.write(frame)
            cv2.imshow("Preview",frame)
            cv2.waitKey(1000//50)
        '''



if __name__ == "__main__":
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Input video to segmet")
    args = parser.parse_args()

    for video_path in glob.glob(os.path.join(args.video,"*.mkv")):
        print(f"Processing {video_path}")
        vs = VideoSegmenter(video_path, show_debug=True, output_size=224)
        frames = vs.get_frames()
        for frame in frames:
            #cv2.imwrite("image.png",frame)
            #break
            pass
        break