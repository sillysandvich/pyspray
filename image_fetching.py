import os.path
import sys

import cv2
import requests
from PIL import Image

#Get url of image source from user input
DUMMY_HEADER = {"User-Agent": "BALLS"}
#Get image(s) from url
#TODO optimize for speed, possibly using ffmpeg
def get_images(path: str, seconds_per_frame: float = 0.2, time_start: float = 0, time_end: float = sys.float_info.max) -> list[Image.Image]:
    #TODO: if switching to ffmpeg, refactor for same function to be used for all animated formats
    def split_animated_image_object(img: Image.Image, seconds_per_frame: float, time_start: float, time_end: float) -> list[Image.Image]:
        extracted_frames = []
        current_time = 0
        frame_start_time = 0
        frame_end_time = 0
        for frame_number in range(img.n_frames):
            img.seek(frame_number)
            frame_duration = img.info['duration'] / 1000 or frame_duration #Last frame duration info sometimes absent for GIFs due to PIL bug
            frame_start_time = frame_end_time
            frame_end_time += frame_duration
            while frame_start_time <= current_time < frame_end_time: 
                if current_time >= time_start:
                    extracted_frames.append(img.copy())
                current_time += seconds_per_frame
                if current_time > time_end: 
                    return extracted_frames
        return extracted_frames
    def split_path_using_cv2(url: str, seconds_per_frame: float, time_start: float, time_end: float) -> list[Image.Image]:
        capture = cv2.VideoCapture(url)
        images = []
        current_time = 0
        frame_start_time = 0
        frame_end_time = 0
        frame_duration = 1 / capture.get(cv2.CAP_PROP_FPS)

        while True:
            capture.grab()
            frame_start_time = frame_end_time
            frame_end_time += frame_duration
            if frame_start_time <= current_time < frame_end_time:
                success, frame = capture.retrieve()
                if not success: break
                while frame_start_time <= current_time < frame_end_time:
                    if current_time >= time_start:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB")
                        images.append(image)
                    current_time += seconds_per_frame
                    if current_time > time_end: 
                        return images 
        return images
    def open_path(path: str, is_local_file: bool) -> Image.Image:
        if is_local_file:
            img = Image.open(path)
        else:
            with requests.get(path, stream = True, headers = DUMMY_HEADER) as response:
                img = Image.open(response.raw)
        return img
    if os.path.isfile(path): #Get resource from local file
        is_local_file = True
        _, ext = os.path.splitext(path)
        file_ending = ext.split(".")[-1]
    else:
        is_local_file = False
        file_ending = path.split("?")[0].split(".")[-1]
    file_ending = file_ending.lower()
    match file_ending:
        case "jpg" | "jpeg":
            img = open_path(path, is_local_file)
            return [img]
        case "png":
            png = open_path(path, is_local_file)
            if png.is_animated:
                return split_animated_image_object(png, seconds_per_frame, time_start, time_end)
            else:
                return [png]
        case "gif":
            gif = open_path(path, is_local_file)
            return split_animated_image_object(gif, seconds_per_frame, time_start, time_end)
        case "webm" | "mp4":
            return split_path_using_cv2(path, seconds_per_frame, time_start, time_end)
        case _:
            raise ValueError(f"Unknown data type, file ending '{file_ending}'")