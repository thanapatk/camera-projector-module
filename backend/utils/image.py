import cv2
import base64
import numpy as np


def base64_to_cv2_image(base64_encoded_img: str):
    image_data = base64.b64decode(base64_encoded_img)

    np_array = np.frombuffer(image_data, np.uint8)

    cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return cv_image
