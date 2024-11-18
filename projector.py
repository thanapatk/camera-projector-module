import numpy as np
import cv2
import os
import threading
import time
from camera.camera import Camera
from motor_controller.motor_controller import StepperController
from utils.singleton import singleton
from collections import deque
from typing import Tuple


@singleton
class Projector:
    def __init__(
        self,
        stepper_controller: StepperController,
        camera_controller: Camera,
        window_name: str,
        fps: int = 30,
        output_size: Tuple[int, int] = (1920, 1080),
    ):
        os.environ["DISPLAY"] = ":0"

        self.fps = fps
        self.frame_time = 1 / fps
        self.window_name = window_name

        self.stepper_controller = stepper_controller
        self.camera_controller = camera_controller

        self.empty_frame = np.zeros((*output_size[::-1], 3), dtype=np.uint8)
        self.video_queue = deque()
        self.freeze_frame = False

        self.video_queue_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

    def display_frames(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(
        #     self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        # )
        while not self.stop_event.is_set():
            start_frame_time = time.perf_counter()
            with self.video_queue_lock:
                if len(self.video_queue) == 0:
                    frame = self.empty_frame
                elif self.freeze_frame:
                    frame = self.video_queue[0]
                else:
                    frame = self.video_queue.popleft()

                cv2.imshow("Auto Focus Projector", frame)

                delta_frame_time = time.perf_counter() - start_frame_time

                if delta_frame_time > self.frame_time:
                    continue

                cv2.waitKey(round(delta_frame_time * 1e3))

    def add_frame(self, frame: np.ndarray):
        with self.video_queue_lock:
            self.video_queue.append(frame)

    def stop(self):
        self.stop_event.set()
        self.display_thread.join()
        cv2.destroyAllWindows()

    def calibrate_focus(self):
        found_four_marker = 0
