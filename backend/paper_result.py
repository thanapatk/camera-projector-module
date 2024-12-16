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
from utils.image import base64_to_cv2_image
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


def check_rect_settled(
    prev_state,
    current_state,
    position_threshold=3.0,
    angle_threshold=3.3,
    ratio_threshold=3.0,
):
    if prev_state is None or current_state is None:
        return False

    prev_center = np.array(prev_state[:2])
    current_center = np.array(current_state[:2])

    prev_ratio = prev_state[2] / prev_state[3]
    current_ratio = current_state[2] / current_state[3]

    prev_angle = prev_state[4]
    current_angle = current_state[4]

    d_pos = np.linalg.norm(current_center - prev_center)
    d_ratio = abs(current_ratio - prev_ratio)
    d_angle = abs(current_angle - prev_angle)

    return (
        d_pos < position_threshold
        and d_ratio < ratio_threshold
        and d_angle < angle_threshold
    )


def run_projector(projector_started, to_ws, from_ws):
    scene_depth = None
    matrix = None
    (x, y, w, h) = [None] * 4

    rect_kf = KalmanFilter()

    depth_threadhold = 20  # mm

    object_depths = deque(maxlen=5)

    uploaded_img = None
    prev_state = None

    detecting_object = False
    is_projecting = False

    projected_img_tx, projected_img_ty, projected_img_scale_x, projected_img_scale_y = [
        None
    ] * 4

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

    with open(f"paper_result_{scene_depth}.csv", "w") as f:
        f.write(
            "object_depth,motor_step,tx,ty,scale_bias,center_x,center_y,scale,width,height"
        )

    while True:
        # handle commands from ws
        if not from_ws.empty():
            cmd = from_ws.get().split(" ")

            if cmd[0] == "upload_image":
                uploaded_img = base64_to_cv2_image(
                    cmd[1].replace("data:image/png;base64,", "")
                )

                to_ws.put("uploaded_image")
            elif cmd[0] == "project":
                (
                    projected_img_tx,
                    projected_img_ty,
                    projected_img_scale_x,
                    projected_img_scale_y,
                ) = map(float, cmd[1:])
                is_projecting = True

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

            detecting_object = False
            prev_state = None
            is_projecting = False
            continue

        # Update motor position according to object depth
        # if len(object_depths) == 0:
        projector.move_to_focus(object_depth)
        # else:
        #     avg_object_depth = np.mean(object_depths, dtype=float)
        #     if not (avg_object_depth - 2 <= object_depth <= avg_object_depth + 2):
        #         projector.move_to_focus(object_depth)

        # object_depths.append(object_depth)

        # Find contour of the closest plane
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # get the largest contour by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]

            rect = cv2.minAreaRect(largest_contour)
            rect = rect_kf.stabilize_angle(rect)

            # Transform the rect from camera's reference frame to projector's reference frame
            center, dimensions, angle, transform_prediction = projector.transform_rect(
                scene_depth, object_depth, rect, matrix
            )

            # Get the smoothed rect from Kalman filter and update its state
            predicted_state = rect_kf.predict()
            rect_kf.update([*center, *dimensions, angle])

            predicted_rect = (
                tuple(predicted_state[:2]),
                tuple(predicted_state[2:4]),
                predicted_state[4],
            )
            predicted_rect = rect_kf.stabilize_angle(predicted_rect)

            is_currently_settled = check_rect_settled(prev_state, predicted_state)

            if not is_currently_settled and not detecting_object:
                to_ws.put("detecting_object")
                detecting_object = True
                is_projecting = False
            elif is_currently_settled and detecting_object:
                to_ws.put(
                    f"detected_ratio {predicted_rect[1][0] / predicted_rect[1][1]}"
                )
                detecting_object = False

            prev_state = predicted_state

            box = np.int_(cv2.boxPoints(predicted_rect))

            output_img = projector.empty_frame_no_border.copy()

            if is_projecting and not (
                uploaded_img is None
                or projected_img_tx is None
                or projected_img_ty is None
                or projected_img_scale_x is None
                or projected_img_scale_y is None
            ):
                projected_img = np.zeros(
                    (int(predicted_rect[1][0]), int(predicted_rect[1][1]), 3),
                    dtype=np.uint8,
                )
                projected_height, projected_width = projected_img.shape[:2]

                # uploaded_img_h, uploaded_img_w = uploaded_img.shape[:2]
                scaled_dim = (
                    int(projected_width * projected_img_scale_x),
                    int(projected_height * projected_img_scale_y),
                )
                scaled_uploaded_img = cv2.resize(uploaded_img, scaled_dim)

                scaled_img_height, scaled_img_width = scaled_uploaded_img.shape[:2]

                tx = int(projected_width * projected_img_tx)
                ty = int(projected_width * projected_img_ty)

                # Calculate placement bounds
                src_x_start = max(0, -tx)  # Start of the source image if tx is negative
                src_y_start = max(0, -ty)  # Start of the source image if ty is negative
                dst_x_start = max(0, tx)  # Start of the destination if tx is positive
                dst_y_start = max(0, ty)  # Start of the destination if ty is positive

                # Calculate the width and height to be copied
                copy_width = min(
                    scaled_img_width - src_x_start, projected_width - dst_x_start
                )
                copy_height = min(
                    scaled_img_height - src_y_start,
                    projected_height - dst_y_start,
                )

                # Ensure no negative dimensions
                if copy_width > 0 and copy_height > 0:
                    # Copy the region from the scaled image to the projected image
                    projected_img[
                        dst_y_start : dst_y_start + copy_height,
                        dst_x_start : dst_x_start + copy_width,
                    ] = scaled_uploaded_img[
                        src_y_start : src_y_start + copy_height,
                        src_x_start : src_x_start + copy_width,
                    ]

                projected_h, projected_w = projected_img.shape[:2]
                src_points = np.array(
                    [
                        [0, 0],  # Top-left
                        [0, projected_h - 1],  # Bottom-left
                        [projected_w - 1, 0],  # Top-right
                        [projected_w - 1, projected_h - 1],  # Bottom-right
                    ],
                    dtype=np.float32,
                )

                # Sort dst_points based on y-coordinate, then x-coordinate
                dst_points = np.array(box, dtype=np.float32)
                sorted_indices = np.lexsort(
                    (dst_points[:, 0], dst_points[:, 1])
                )  # Sort by y, then x
                sorted_points = dst_points[sorted_indices]

                # Top two and bottom two points
                top_points = sorted_points[:2]
                bottom_points = sorted_points[2:]

                # Sort top and bottom points by x-coordinate
                top_left, top_right = top_points[np.argsort(top_points[:, 0])]
                bottom_left, bottom_right = bottom_points[
                    np.argsort(bottom_points[:, 0])
                ]

                # Reorder dst_points
                dst_points_ordered = np.array(
                    [top_left, bottom_left, top_right, bottom_right], dtype=np.float32
                )

                transform_matrix = cv2.getPerspectiveTransform(
                    src_points, dst_points_ordered
                )

                warped_img = cv2.warpPerspective(
                    projected_img,
                    transform_matrix,
                    (output_img.shape[1], output_img.shape[0]),
                )

                output_img = cv2.addWeighted(output_img, 1, warped_img, 1, 0)

            else:
                cv2.drawContours(
                    output_img,
                    [box],
                    -1,
                    (0, 255, 0) if is_currently_settled else (0, 0, 255),
                    2,
                )

            projector.add_frame(output_img)
            with open("paper_result.csv", "a+") as f:
                # object_depth,motor_step,tx,ty,scale_bias,center_x,center_y,scale,width,height
                f.write(
                    f"{object_depth},{controller.step},{','.join(map(str, transform_prediction))},{','.join(map(str, predicted_rect[0]))}"
                )


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
