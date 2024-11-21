import configparser

from numpy import size
from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector import Projector
from time import sleep

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
    # home_motor=False,
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
    fps=camera_config.getint("fps"),
)

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    window_name="Auto Focus Projector",
    # output_size=(1280, 720),
)

freq = 1000 * 2  # freq to move to limit 225

if __name__ == "__main__":
    # controller.move_to_step(controller.step_range, 225 * 2)

    depth, step, area = projector.calibrate_focus()

    print(f"Depth: {depth}")
    print(f"Step: {step}")
    print(f"Area: {area}")

    projector.stop()
    camera.stop()
    controller.stop()
