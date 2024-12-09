import numpy as np


class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(
            5
        )  # State vector: [center_x, center_y, width, height, angle]

        self.P = np.eye(5)  # State covariance matrix

        self.F = np.eye(5)  # Transition matrix (state model)

        self.H = np.eye(5)  # Measurement matrix (what we observe)

        self.Q = np.eye(5) * 0.05  # Process noise covariance

        self.R = np.eye(5) * 1  # Measurement noise covariance

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

    @staticmethod
    def stabilize_angle(rect):
        center, size, angle = rect
        width, height = size

        # Normalize: Ensure width >= height
        if width < height:
            width, height = height, width
            angle += 90  # Adjust the angle when swapping dimensions

        # Normalize angle to [-90, 90)
        angle %= 360

        return (center, (width, height), angle)
