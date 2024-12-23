import configparser
import os

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
    home_motor=False,
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

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    window_name="Auto Focus Projector",
    focal_length=config["Lens"].getfloat("f"),
)

if __name__ == "__main__":
    for i in range(5):
        controller.home_motor()

        depth, step = projector.calibrate_focus()

        print(f"Depth: {depth} mm")
        print(f"Step: {step}")

        output_file_exists = os.path.isfile("calibration_result.csv")

        with open("calibration_result.csv", "a+") as f:
            if not output_file_exists:
                f.write("depth,step\n")
            f.write(",".join(map(str, [depth, step])) + "\n")

    projector.stop()
    camera.stop()
    controller.stop()
