import cv2
from cv2 import aruco
import numpy as np

GRID_SIZE = 90
MARKER_SIZE = round(GRID_SIZE * 2)
GRID_COUNT = (13, 7)
SCREEN_SIZE = (1920, 1080)
PADDING = (
    (np.array(SCREEN_SIZE) - np.array(GRID_COUNT) * GRID_SIZE) // 2
    - np.array((MARKER_SIZE, MARKER_SIZE))
) // 2

markers = [np.zeros((MARKER_SIZE, MARKER_SIZE), dtype=np.uint8) for _ in range(4)]
markers = [
    aruco.drawMarker(
        aruco.Dictionary_get(cv2.aruco.DICT_6X6_50),
        id,
        MARKER_SIZE,
        img,
        1,
    )
    for id, img in enumerate(markers)
]

img = np.ones(SCREEN_SIZE[::-1], dtype=np.uint8) * 255

top_left = np.array(SCREEN_SIZE) // 2 - np.array(GRID_COUNT) * GRID_SIZE // 2
black_tiles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
for j in range(GRID_COUNT[1]):
    for i in range(GRID_COUNT[0]):
        if (i + j) % 2 == 0:
            continue
        current_top_left = top_left + np.array((i, j)) * GRID_SIZE
        img[
            current_top_left[1] : current_top_left[1] + GRID_SIZE,
            current_top_left[0] : current_top_left[0] + GRID_SIZE,
        ] = black_tiles

for i, marker in enumerate(markers):
    pos = (
        (PADDING[0], SCREEN_SIZE[0] - PADDING[0] - MARKER_SIZE)[i % 2],
        (PADDING[1], SCREEN_SIZE[1] - PADDING[1] - MARKER_SIZE)[i // 2],
    )

    img[pos[1] : pos[1] + MARKER_SIZE, pos[0] : pos[0] + MARKER_SIZE] = marker


cv2.imwrite("calibration1.png", img)
