import pyrealsense2 as rs
from typing import Tuple
from utils.singleton import singleton


class CameraConfig:
    CALIBRATION_MODE = {
        rs.option.enable_auto_exposure: False,
        rs.option.exposure: 80.0,
        rs.option.gain: 7,
        rs.option.brightness: -64.0,
        rs.option.contrast: 100.0,
        rs.option.gamma: 300.0,
    }
    NORMAL_MODE = {
        rs.option.enable_auto_exposure: True,
        rs.option.brightness: 0.0,
        rs.option.contrast: 50.0,
        rs.option.gamma: 300.0,
    }


@singleton
class Camera:
    def __init__(self, size: Tuple[int, int], fps: int):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.size = size
        self.fps = fps

        self.config.enable_stream(rs.stream.color, *self.size, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, *self.size, rs.format.z16, self.fps)

        self.queue = rs.frame_queue(50, keep_frames=True)

        self.profile = self.pipeline.start(self.config, self.queue)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.color_sensor = self.profile.get_device().first_color_sensor()

        self.apply_config(CameraConfig.NORMAL_MODE)

        self.align = rs.align(rs.stream.color)

    def stop(self):
        self.pipeline.stop()

    def apply_config(self, config: dict):
        for key, value in config.items():
            self.color_sensor.set_option(key, value)

    def get_frames(self):
        """
        depth_frame, color_frame
        """
        frames = self.queue.wait_for_frame()

        aligned_frames = self.align.process(frames.as_frameset())

        return aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()


# if __name__ == "__main__":
#     import cv2
#     import numpy as np
#
#     __import__("os").environ["DISPLAY"] = ":0"
#
#     camera = Camera(size=(640, 480), fps=30)
#
#     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
#     aruco_params = cv2.aruco.DetectorParameters_create()
#
#     try:
#         while True:
#             _, color_frame = camera.get_frames()
#
#             if not color_frame:
#                 continue
#
#             color_img = np.asanyarray(color_frame.get_data())
#
#             corners, ids, rejected = cv2.aruco.detectMarkers(
#                 color_img, aruco_dict, parameters=aruco_params
#             )
#
#             new_img = cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
#
#             cv2.imshow("color frame", color_img)
#             cv2.imshow("markers", new_img)
#
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#
#     except Exception as e:
#         print(e)
#     finally:
#         cv2.destroyAllWindows()
