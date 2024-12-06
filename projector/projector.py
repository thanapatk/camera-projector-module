import numpy as np
import cv2
import os
import threading
import time

from camera.camera import Camera, CameraConfig
from motor_controller.motor_controller import StepperController
from utils.singleton import singleton
from collections import deque
from typing import Optional, Tuple


@singleton
class Projector:
    __DESIRED_CORNERS = np.array(
        ((97, 22), (1822, 22), (1822, 1057), (97, 1057)),
        dtype=np.float32,
    )
    __DESIRED_WIDTH = 1822 - 97

    def __init__(
        self,
        stepper_controller: StepperController,
        camera_controller: Camera,
        focal_length: float,
        window_name: str,
        fps: int = 30,
        output_size: Tuple[int, int] = (1920, 1080),
    ):
        os.environ["DISPLAY"] = ":0"

        self.fps = fps
        self.frame_time = 1 / fps
        self.window_name = window_name
        self.focal_length = focal_length

        self.stepper_controller = stepper_controller
        self.camera_controller = camera_controller

        self.output_size = output_size

        self.empty_frame_no_border = np.zeros(
            (*self.output_size[::-1], 3), dtype=np.uint8
        )
        self.empty_frame = self.empty_frame_no_border.copy()
        line_thickness = 2
        cv2.rectangle(
            self.empty_frame,
            (line_thickness, line_thickness),
            (
                self.output_size[0] - line_thickness,
                self.output_size[1] - line_thickness,
            ),
            (0, 0, 255),
            line_thickness,
        )

        self.video_queue = deque()
        self.freeze_frame = False
        self.hold_last_frame = True

        self.video_queue_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

    def display_frames(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        while not self.stop_event.is_set():
            start_frame_time = time.perf_counter()
            with self.video_queue_lock:
                if len(self.video_queue) == 0:
                    frame = self.empty_frame
                elif self.freeze_frame or (
                    len(self.video_queue) == 1 and self.hold_last_frame
                ):
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

    @staticmethod
    def get_center(corners: np.ndarray) -> np.ndarray:
        return (corners[0] + corners[2]) / 2

    @staticmethod
    def get_corners(
        corners,
        ids,
        corners_loc: Optional[Tuple[int, int, int, int]] = None,
        dtype: type = np.float32,
    ) -> np.ndarray:
        detected_corners = np.empty((4, 2), dtype=dtype)

        for corner, id in zip(corners, ids):
            index = id[0]
            if corners_loc:
                corner_index = corners_loc[index]
            else:
                corner_index = index
            detected_corners[index] = corner[0][corner_index]

        return detected_corners

    @classmethod
    def normalized_corners(cls, corners: np.ndarray) -> np.ndarray:
        x_coords = corners[:, 0]

        x_min = np.min(x_coords)
        x_max = np.max(x_coords)

        return corners / (x_max - x_min) * cls.__DESIRED_WIDTH

    @classmethod
    def get_relative_corners(cls, corners: np.ndarray) -> np.ndarray:
        center = cls.get_center(corners)

        return corners - center

    @staticmethod
    def bounding_rect_from_corners(corners: np.ndarray) -> Tuple[int, int, int, int]:
        x_min, y_min = np.min(corners, axis=0).ravel()
        x_max, y_max = np.max(corners, axis=0).ravel()

        output = x_min, y_min, x_max - x_min, y_max - y_min

        return tuple(map(int, output))

    @classmethod
    def find_align_matrix(
        cls,
        corners: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ids: np.ndarray,
        relative: bool = True,
        normalize: bool = True,
    ):
        detected_corners = cls.get_corners(corners, ids)
        desired_corners = cls.__DESIRED_CORNERS

        if relative:
            desired_corners = cls.get_relative_corners(desired_corners)
            detected_corners = cls.get_relative_corners(detected_corners)
        if normalize:
            detected_corners = cls.normalized_corners(detected_corners)

        H, _ = cv2.findHomography(detected_corners, desired_corners)

        return H

    def create_aligned_image(self, img: np.ndarray, warp_matrix):
        h, w = img.shape[:2]

        corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32).reshape(
            -1, 1, 2
        )
        warped_corners = cv2.perspectiveTransform(corners, warp_matrix)

        x_min, y_min = warped_corners.min(axis=0).ravel() - 0.5
        x_max, y_max = warped_corners.max(axis=0).ravel() + 0.5

        translate_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        size = (round(x_max - x_min), round(y_max - y_min))

        self.auto_keystone_matrix = translate_matrix @ warp_matrix
        corrected_image = cv2.warpPerspective(img, self.auto_keystone_matrix, size)

        scale = min(np.array(self.output_size) / size)
        scaled_size = (int(size[0] * scale), int(size[1] * scale))

        resized_image = cv2.resize(corrected_image, scaled_size)

        scene = np.zeros((*self.output_size[::-1], 3), np.uint8)

        top_left = (np.array(scene.shape[:2]) - np.array(resized_image.shape[:2])) // 2

        scene[
            top_left[0] : top_left[0] + resized_image.shape[0],
            top_left[1] : top_left[1] + resized_image.shape[1],
        ] = resized_image

        return scene

    @staticmethod
    def calculate_michelson_contrast(image: np.ndarray) -> float:
        """
        Calculates the Michelson contrast of a checkerboard pattern.

        Args:
            image (numpy.ndarray): image of the checkerboard pattern.

        Returns:
            float: Michelson contrast value.
        """
        # Ensure the image is in gray scale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Otsu's threshold to segment the bright and dark regions
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Separate bright and dark regions using the thresholded mask
        bright_region = gray[thresh == 255]
        dark_region = gray[thresh == 0]

        # Calculate I_max and I_min for Michelson contrast
        I_max = bright_region.mean() if bright_region.size > 0 else 0
        I_min = dark_region.mean() if dark_region.size > 0 else 0

        # Calculate Michelson contrast
        if I_max + I_min == 0:
            return 0  # Avoid division by zero

        contrast = (I_max - I_min) / (I_max + I_min)
        return contrast

    def calibrate_focus(self):
        found_four_markers = False

        calibration_img = cv2.imread("camera/calibration.png")
        with self.video_queue_lock:
            self.freeze_frame = True
            self.video_queue.append(cv2.resize(calibration_img, self.output_size))

        self.camera_controller.apply_config(CameraConfig.CALIBRATION_MODE)

        while not found_four_markers:
            try:
                self.stepper_controller.increment_degree()
            except ValueError:
                raise Exception("Cannot find markers")

            _, color_frame = self.camera_controller.get_frames()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            corners, ids, _ = cv2.aruco.detectMarkers(
                color_img, self.aruco_dict, parameters=self.aruco_params
            )

            if len(corners) == 4:
                found_four_markers = True

        _, aligned_calibration_img = self.auto_keystone(calibration_img)

        with self.video_queue_lock:
            self.video_queue.popleft()
            self.video_queue.append(aligned_calibration_img)

        time.sleep(0.5)

        while True:
            depth_frame, color_frame = self.camera_controller.get_frames()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            corners, ids, _ = cv2.aruco.detectMarkers(
                color_img, self.aruco_dict, parameters=self.aruco_params
            )

            if len(corners) == 4:
                ROI_corners = self.get_corners(corners, ids, corners_loc=(2, 3, 1, 0))
                break

        x, y, w, h = self.bounding_rect_from_corners(ROI_corners)
        x += int(w * 0.025)
        w = int(w * 0.95)
        y += int(h * 0.025)
        h = int(h * 0.95)

        depth_data = np.asanyarray(depth_frame.get_data())
        depth_ROI = depth_data[y : y + h, x : x + w]
        average_depth = np.mean(depth_ROI[depth_ROI != 0])

        found_max_contrast = False
        max_contrast = -1
        max_step = 0

        self.stepper_controller.decrement_degree(degree=20)
        time.sleep(0.5)

        while not found_max_contrast:
            _, color_frame = self.camera_controller.get_frames()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            ROI = color_img[y : y + h, x : x + w]

            contrast = self.calculate_michelson_contrast(ROI)

            if contrast > max_contrast:
                max_contrast = contrast
                max_step = self.stepper_controller.step
            elif abs(max_contrast - contrast) > 0.1:
                break

            try:
                self.stepper_controller.increment_degree()
            except ValueError:
                break

        self.stepper_controller.move_to_step(max_step)
        time.sleep(0.5)

        while True:
            _, color_frame = self.camera_controller.get_frames()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            corners, ids, _ = cv2.aruco.detectMarkers(
                color_img, self.aruco_dict, parameters=self.aruco_params
            )

            if len(corners) == 4:
                outer_corners = self.get_corners(corners, ids)
                area = cv2.contourArea(outer_corners)
                break

        return average_depth, max_step, area

    @staticmethod
    def get_approx_step(depth: float) -> int:
        return int(-0.0042 * depth**2 + 7.2565 * depth - 1911.1)

    def transform_rect(
        self,
        scene_depth: float,
        object_depth: float,
        rect: Tuple[Tuple[float, float], Tuple[float, float], float],
        matrix: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Apply scaling and a homography transformation to a rotated rectangle.

        Args:
            scene_depth (float): Depth of the scene.
            object_depth (float): Depth of the object.
            rect (Tuple[Tuple[float, float], Tuple[float, float], float]): The rectangle in (center, (width, height), angle) format.
            matrix (np.ndarray): The homography matrix to apply.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float], float]: The transformed rectangle in (center, (width, height), angle) format.
        """
        tx, ty = self.get_translation_vector(scene_depth, object_depth)

        scale = (
            (object_depth - self.focal_length)
            / (scene_depth - self.focal_length)
            * (scene_depth / object_depth)
            * bias
        )

        # Scale the rectangle dimensions
        new_rect = cv2.boxPoints(rect)
        new_rect = np.array(new_rect, dtype="float32")

        # Step 2: Apply homography to each corner
        ones = np.ones(
            (new_rect.shape[0], 1), dtype="float32"
        )  # Add ones for homogeneous coordinates
        rect_homogeneous = np.hstack([new_rect, ones])  # Shape: (4, 3)

        # Transform using the combined transformation matrix
        transformed_corners = (matrix @ rect_homogeneous.T).T  # Shape: (4, 3)

        # Normalize by the last coordinate (prevent division by zero)
        transformed_corners[:, :2] /= np.maximum(
            transformed_corners[:, 2].reshape(-1, 1), 1e-5
        )

        # Step 3: Convert back to the rotated rectangle representation
        transformed_points = transformed_corners[:, :2].astype(
            "float32"
        )  # Extract (x, y)

        if (
            transformed_points.shape[0] >= 3
        ):  # At least 3 points required for minAreaRect
            transformed_rect = cv2.minAreaRect(transformed_points)
        else:
            raise ValueError(
                "Insufficient points for minAreaRect after transformation."
            )

        (x, y), (width, height), angle = transformed_rect

        return ((x + tx, y + ty), (width * scale, height * scale), angle)

    def transform_rect_calibration(
        self,
        scene_depth: float,
        object_depth: float,
        rect: Tuple[Tuple[float, float], Tuple[float, float], float],
        matrix: np.ndarray,
        tx: float,
        ty: float,
        bias: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Apply scaling and a homography transformation to a rotated rectangle.

        Args:
            scene_depth (float): Depth of the scene.
            object_depth (float): Depth of the object.
            rect (Tuple[Tuple[float, float], Tuple[float, float], float]): The rectangle in (center, (width, height), angle) format.
            matrix (np.ndarray): The homography matrix to apply.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float], float]: The transformed rectangle in (center, (width, height), angle) format.
        """
        # Calculate depth-dependent scale
        scale = (
            (object_depth - self.focal_length)
            / (scene_depth - self.focal_length)
            * (scene_depth / object_depth)
            * bias
        )

        # Scale the rectangle dimensions
        new_rect = cv2.boxPoints(rect)
        new_rect = np.array(new_rect, dtype="float32")

        # Step 2: Apply homography to each corner
        ones = np.ones(
            (new_rect.shape[0], 1), dtype="float32"
        )  # Add ones for homogeneous coordinates
        rect_homogeneous = np.hstack([new_rect, ones])  # Shape: (4, 3)

        # Transform using the combined transformation matrix
        transformed_corners = (matrix @ rect_homogeneous.T).T  # Shape: (4, 3)

        # Normalize by the last coordinate (prevent division by zero)
        transformed_corners[:, :2] /= np.maximum(
            transformed_corners[:, 2].reshape(-1, 1), 1e-5
        )

        # Step 3: Convert back to the rotated rectangle representation
        transformed_points = transformed_corners[:, :2].astype(
            "float32"
        )  # Extract (x, y)

        if (
            transformed_points.shape[0] >= 3
        ):  # At least 3 points required for minAreaRect
            transformed_rect = cv2.minAreaRect(transformed_points)
        else:
            raise ValueError(
                "Insufficient points for minAreaRect after transformation."
            )

        (x, y), (width, height), angle = transformed_rect

        return ((x + tx, y + ty), (width * scale, height * scale), angle)

    def move_to_focus(self, depth: float):
        self.stepper_controller.move_to_step(self.get_approx_step(depth))

    def auto_keystone(self, img: np.ndarray):
        while True:
            _, color_frame = self.camera_controller.get_frames()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            corners, ids, _ = cv2.aruco.detectMarkers(
                color_img, self.aruco_dict, parameters=self.aruco_params
            )

            if len(corners) == 4:
                break

        warp_matrix = self.find_align_matrix(corners, ids)

        return warp_matrix, self.create_aligned_image(img, warp_matrix)

    @staticmethod
    def apply_matrix(img, matrix):
        return cv2.warpPerspective(img, matrix, img.shape[:2][::-1])
