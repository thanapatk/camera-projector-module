import cv2
import configparser
import numpy as np
import time
import re
from collections import deque
from ctypes import c_bool
from multiprocessing import Process, Value, Queue

from backend import run_fastapi
from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector.projector import Projector
from utils.contour_transformer import ContourTransformationModels
from utils.kalman_filer import KalmanFilter


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
    color_fps=camera_config.getint("depth_fps"),
    depth_fps=camera_config.getint("depth_fps"),
)

with open("models.pkl", "rb") as f:
    contour_transformer = ContourTransformationModels(f)

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    contour_transformer=contour_transformer,
    window_name="Auto Focus Projector",
    focal_length=config["Lens"].getfloat("f"),
)


def run_projector(projector_started, to_ws, from_ws):
    scene_depth = None
    matrix = None
    (x, y, w, h) = [None] * 4

    rect_kf = KalmanFilter()

    depth_threadhold = 20  # mm

    object_depths = deque(maxlen=5)

    while not projector_started.value:
        if not from_ws.empty():
            cmd = from_ws.get()

            if not re.search(r"start_projector_(?:with|no)_calibration", cmd):
                time.sleep(0.5)
                continue

            if cmd == "start_projector_with_calibration":
                scene_depth, matrix, (x, y, w, h) = projector.start_with_calibration()
            elif cmd == "start_projector_no_calibration":
                scene_depth, matrix, (x, y, w, h) = projector.start_no_calibration()

            to_ws.put("started_projector")
            projector_started.value = True

    if (
        scene_depth is None
        or matrix is None
        or x is None
        or y is None
        or w is None
        or h is None
    ):
        return

    while True:
        depth_frame = camera.get_depth_frame()

        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image[y : y + h, x : x + w]  # ROI

        min_depth = np.min(depth_image[depth_image > 0])

        # Get mask of the closest plane to the projector
        binary_mask = np.zeros_like(depth_image, dtype=np.uint8)
        binary_mask[
            (depth_image >= min_depth) & (depth_image <= min_depth + depth_threadhold)
        ] = 255

        object_depth = np.mean(depth_image[binary_mask == 255])

        # Don't continue if closest plane is near scene_depth (Assumed as depth noise)
        if (
            scene_depth - depth_threadhold / 2
            <= object_depth
            <= scene_depth + depth_threadhold / 2
        ):
            projector.move_to_focus(scene_depth)
            projector.add_frame(projector.empty_frame)
            object_depths.clear()
            continue

        # Update motor position according to object depth
        if len(object_depths) == 0:
            projector.move_to_focus(object_depth)
        else:
            avg_object_depth = np.mean(object_depths, dtype=float)
            if not (avg_object_depth - 2 <= object_depth <= avg_object_depth + 2):
                projector.move_to_focus(object_depth)

        object_depths.append(object_depth)

        # Find contour of the closest plane
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # get the largest contour by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]

            rect = cv2.minAreaRect(largest_contour)

            # Transform the rect from camera's reference frame to projector's reference frame
            center, dimensions, angle = projector.transform_rect(
                scene_depth, np.mean(object_depths, dtype=float), rect, matrix
            )

            # Get the smoothed rect from Kalman filter and update its state
            predicted_state = rect_kf.predict()
            rect_kf.update([*center, *dimensions, angle])

            predicted_rect = (
                tuple(predicted_state[:2]),
                tuple(predicted_state[2:4]),
                predicted_state[4],
            )

            box = np.int_(cv2.boxPoints(predicted_rect))

            output_img = projector.empty_frame_no_border.copy()
            cv2.drawContours(output_img, [box], -1, (0, 255, 0), 2)

            projector.add_frame(output_img)


if __name__ == "__main__":
    projector_started = Value(c_bool, False)
    to_ws = Queue()
    from_ws = Queue()

    fastapi_process = Process(
        target=run_fastapi, args=(projector_started, to_ws, from_ws)
    )
    # projector_process = Process(
    #     target=run_projector, args=(projector_started, to_ws, from_ws)
    # )

    fastapi_process.start()
    # projector_process.start()

    # wait for keyboard interrupt
    try:
        run_projector(projector_started, to_ws, from_ws)
    except KeyboardInterrupt:
        print("Recieved Keyboard Interrupt")
    finally:
        print("Terminating processes...")

        # Gracefully terminate child processes
        fastapi_process.terminate()
        # projector_process.terminate()

        # Wait for child processes to clean up
        fastapi_process.join()
        # projector_process.join()

        # Stop hardware controllers and other resources
        projector.stop()
        camera.stop()
        controller.stop()

        print("Processes terminated successfully.")
