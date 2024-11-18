import configparser

from numpy import size
from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector import Projector
from cv2 import imread
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
    size=(camera_config.getint("width"), camera_config.getint("height")),
    fps=camera_config.getint("fps"),
)

projector = Projector(
    stepper_controller=controller,
    camera_controller=camera,
    window_name="Auto Focus Projector",
    output_size=(640, 480),
)

freq = 1000 * 2  # freq to move to limit 225

if __name__ == "__main__":
    projector.freeze_frame = True
    projector.add_frame(imread("camera/calibration.png"))

    # sleep(10)

    # for i in range(10):
    #     controller.move_to_step(controller.step_range * 3 // 4, freq)
    #     controller.move_to_step(controller.step_range // 4, freq)

    controller.move_to_step(controller.step_range, 225 * 2)

    # projector.stop_event.set()

    projector.stop()
    camera.stop()
    controller.stop()
