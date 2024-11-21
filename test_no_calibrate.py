import configparser
import numpy as np
import cv2
import time

from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector import Projector

config = configparser.ConfigParser()
config.read("config.ini")

stepper_config = config["Stepper Motor"]
degree_range = (
    stepper_config.getint("output degree")
    * stepper_config.getint("output teeth")
    / stepper_config.getint("input teeth")
)

controller = StepperController(
    step_angle=stepper_config.getfloat("step angle"),
    micro_stepping=stepper_config.getint("micro stepping"),
    degree_range=degree_range,
    pins=StepperPins(
        step_pin=stepper_config.getint("step pin"),
        dir_pin=stepper_config.getint("dir pin"),
        enabled_pin=stepper_config.getint("en pin"),
    ),
)

camera_config = config["Camera"]
camera = Camera(
    color_size=(
        camera_config.getint("color_width"),
        camera_config.getint("color_height"),
    ),
    depth_size=(
        camera_config.getint("depth_width"),
        camera_config.getint("depth_height"),
    ),
    fps=camera_config.getint("fps"),
)

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    window_name="Auto Focus Projector",
)

if __name__ == "__main__":
    depth_frame, _ = camera.get_frames()

    if not depth_frame:
        exit()

    depth_data = np.asanyarray(depth_frame.get_data())

    h, w = depth_data.shape[:2]

    depth = depth_data[h // 2, w // 2]

    img = cv2.imread("camera/calibration.png")
    projector.freeze_frame = True
    projector.add_frame(img)

    projector.move_to_focus(depth)
    aligned = projector.auto_keystone(img)

    with projector.video_queue_lock:
        projector.video_queue.append(aligned)
        projector.video_queue.popleft()

    time.sleep(10)

    projector.stop()
    camera.stop()
    controller.stop()
