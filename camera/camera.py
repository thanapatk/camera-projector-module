import pyrealsense2 as rs
from typing import Tuple
from utils.singleton import singleton


class CameraConfig:
    CALIBRATION_MODE = {
        rs.option.enable_auto_exposure: False,
        rs.option.exposure: 100.0,
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
    def __init__(
        self,
        color_size: Tuple[int, int] = (1280, 720),
        depth_size: Tuple[int, int] = (640, 480),
        color_fps: int = 30,
        depth_fps: int = 30,
        decimation_magnitude=1.0,
        spatial_magnitude=2.0,
        spatial_smooth_alpha=0.5,
        spatial_smooth_delta=20,
        temporal_smooth_alpha=0.4,
        temporal_smooth_delta=20,
    ):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.color_size = color_size
        self.depth_size = depth_size

        self.color_fps = color_fps
        self.depth_fps = depth_fps

        self.config.enable_stream(
            rs.stream.color, *self.color_size, rs.format.bgr8, self.color_fps
        )
        self.config.enable_stream(
            rs.stream.depth, *self.depth_size, rs.format.z16, self.depth_fps
        )

        # self.queue = rs.frame_queue(50, keep_frames=True)
        #
        # self.profile = self.pipeline.start(self.config, self.queue)
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.color_sensor = self.profile.get_device().first_color_sensor()

        self.apply_config(CameraConfig.NORMAL_MODE)

        self.align = rs.align(rs.stream.color)

        # Available filters and control options for the filters
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

        # Apply the control parameters for the filter
        self.decimation_filter.set_option(
            rs.option.filter_magnitude, decimation_magnitude
        )
        self.spatial_filter.set_option(rs.option.filter_magnitude, spatial_magnitude)
        self.spatial_filter.set_option(
            rs.option.filter_smooth_alpha, spatial_smooth_alpha
        )
        self.spatial_filter.set_option(
            rs.option.filter_smooth_delta, spatial_smooth_delta
        )
        self.temporal_filter.set_option(
            rs.option.filter_smooth_alpha, temporal_smooth_alpha
        )
        self.temporal_filter.set_option(
            rs.option.filter_smooth_delta, temporal_smooth_delta
        )

    def stop(self):
        self.pipeline.stop()

    def apply_config(self, config: dict):
        for key, value in config.items():
            self.color_sensor.set_option(key, value)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        return (
            self.post_process_depth_frame(aligned_frames.get_depth_frame()),
            aligned_frames.get_color_frame(),
        )

    def post_process_depth_frame(self, depth_frame):
        """
        Filter the depth frame acquired using the Intel RealSense device

        Parameters:
        -----------
        depth_frame          : rs.frame()
                               The depth frame to be post-processed
        decimation_magnitude : double
                               The magnitude of the decimation filter
        spatial_magnitude    : double
                               The magnitude of the spatial filter
        spatial_smooth_alpha : double
                               The alpha value for spatial filter based smoothening
        spatial_smooth_delta : double
                               The delta value for spatial filter based smoothening
        temporal_smooth_alpha: double
                               The alpha value for temporal filter based smoothening
        temporal_smooth_delta: double
                               The delta value for temporal filter based smoothening

        Return:
        ----------
        filtered_frame : rs.frame()
                         The post-processed depth frame
        """

        # Post processing possible only on the depth_frame
        assert depth_frame.is_depth_frame()

        # Apply the filters
        filtered_frame = self.decimation_filter.process(depth_frame)
        filtered_frame = self.spatial_filter.process(filtered_frame)
        filtered_frame = self.temporal_filter.process(filtered_frame)

        return filtered_frame
