from collections import deque
import os
import csv
import configparser
import numpy as np
import cv2

from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector.projector import Projector

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
    color_size=(1280, 720), depth_size=(1280, 720), color_fps=30, depth_fps=30
)

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    window_name="Auto Focus Projector",
    focal_length=config["Lens"].getfloat("f"),
)


class KalmanFilter:
    def __init__(self):
        # State vector: [center_x, center_y, width, height, angle]
        self.state = np.zeros(5)

        # State covariance matrix
        self.P = np.eye(5)

        # Transition matrix (state model)
        self.F = np.eye(5)

        # Measurement matrix (what we observe)
        self.H = np.eye(5)

        # Process noise covariance
        self.Q = np.eye(5) * 0.01

        # Measurement noise covariance
        self.R = np.eye(5) * 1.0

    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        # Update the state with the new measurement
        measurement = np.array(measurement)
        y = measurement - (self.H @ self.state)  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state += K @ y  # Update state estimate
        self.P = (np.eye(len(self.state)) - K @ self.H) @ self.P  # Update covariance


avg_depths = deque(maxlen=5)
rect_kf = KalmanFilter()
tx, ty = 0, 0
bias = 1


def append_data_to_csv(
    csv_file_path: str,
    object_depth: float,
    scene_depth: float,
    tx: float,
    ty: float,
    bias: float,
):
    # Check if the file exists to determine if the header is needed
    file_exists = os.path.exists(csv_file_path)

    row = [
        scene_depth,
        object_depth,
        tx,
        ty,
        bias,
    ]

    # Define the header
    header = [
        "scene_depth",
        "object_depth",
        "tx",
        "ty",
        "bias",
    ]

    # Append data to CSV
    with open(csv_file_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if the file does not already exist
        if not file_exists:
            writer.writerow(header)

        # Write rows
        writer.writerow(row)

    print(f"Calibration data appended to {csv_file_path}")


def main():
    while True:
        depth_frame = camera.get_depth_frame()

        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())

        depth_image = depth_image[y : y + h, x : x + w]
        avg_depth = np.mean(depth_image[depth_image != 0])

        min_depth = np.min(depth_image[depth_image > 0])
        depth_threshold = 20

        binary_mask = np.zeros_like(depth_image, dtype=np.uint8)
        binary_mask[
            (depth_image >= min_depth) & (depth_image <= min_depth + depth_threshold)
        ] = 255
        avg_depth = np.mean(depth_image[binary_mask == 255])

        if scene_depth - 25 <= avg_depth <= scene_depth + 25:
            projector.move_to_focus(scene_depth)
            projector.add_frame(projector.empty_frame)
            continue

        projector.move_to_focus(avg_depth)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            avg_depths.append(avg_depth)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]

            rect = cv2.minAreaRect(largest_contour)

            center, (width, height), angle = projector.transform_rect_calibration(
                scene_depth=scene_depth,
                object_depth=avg_depth,
                rect=rect,
                matrix=matrix,
                tx=tx,
                ty=ty,
                bias=bias,
            )

            smoothed_state = rect_kf.predict()

            rect_kf.update([*center, width, height, angle])

            smoothed_rect = (
                (smoothed_state[0], smoothed_state[1]),  # Smoothed center
                (smoothed_state[2], smoothed_state[3]),  # Smoothed size
                smoothed_state[4],  # Smoothed angle
            )

            center = tuple(map(int, smoothed_rect[0]))

            box = cv2.boxPoints(smoothed_rect)
            box = np.int_(box)
            contour_img = projector.empty_frame.copy()
            cv2.drawContours(contour_img, [box], -1, (0, 255, 0), 2)
            cv2.circle(contour_img, center, 2, (0, 0, 255), -1)

            projector.add_frame(contour_img)


if __name__ == "__main__":
    scene_depth, matrix, (x, y, w, h) = projector.start_no_calibration()
    print(f"Scene Depth: {scene_depth}")

    while True:
        try:
            main()
        except KeyboardInterrupt:
            cmd = input("\nEnter command {[e]xit, [u]pdate, [s]ave}: ")

            if cmd == "exit" or cmd == "e":
                break
            elif cmd == "update" or cmd == "u":
                exec(input())
            elif cmd == "save" or cmd == "s":
                append_data_to_csv(
                    "transformation_calibration.csv",
                    np.mean(avg_depths, dtype=float),
                    scene_depth,
                    tx,
                    ty,
                    bias,
                )
        except Exception as e:
            raise e

    projector.stop()
    camera.stop()
    controller.stop()
