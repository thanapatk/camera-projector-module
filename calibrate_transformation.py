from collections import deque
import os
import csv
import configparser
import numpy as np
import cv2
import time

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
        depth_frame, _ = camera.get_frames()

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

            if len(avg_depths) > 10:
                avg_depths.popleft()

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
    _, aligned = projector.auto_keystone(img)

    with projector.video_queue_lock:
        projector.video_queue.append(aligned)
        projector.video_queue.popleft()

    time.sleep(1)

    _, color_frame = camera.get_frames()

    color_img = np.asanyarray(color_frame.get_data())

    corners, ids, _ = cv2.aruco.detectMarkers(
        color_img, projector.aruco_dict, parameters=projector.aruco_params
    )

    with projector.video_queue_lock:
        projector.freeze_frame = False
        projector.video_queue.clear()

    time.sleep(1)

    matrix = projector.find_align_matrix(corners, ids, relative=True, normalize=False)

    x, y, w, h = projector.bounding_rect_from_corners(
        projector.get_corners(corners, ids)
    )

    # transform to fit whole area
    real_w, real_h = 1822 - 97, 1057 - 22
    x -= int(97 * w / real_w)
    y -= int(22 * h / real_h)
    w = int(w * 1.113)
    h = int(h * 1.043)

    depth_frame, _ = camera.get_frames()
    depth_data = np.asanyarray(depth_frame.get_data())

    depth_ROI = depth_data[y : y + h, x : x + w]
    scene_depth = np.mean(depth_ROI[depth_ROI != 0])
    projector.move_to_focus(scene_depth)

    aligned_empty = projector.apply_matrix(projector.empty_frame, matrix)

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
