import configparser

from camera.camera import Camera
from motor_controller.motor_controller import StepperController, StepperPins
from projector.projector import Projector
from utils.contour_transformer import ContourTransformationModels

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

if __name__ == "__main__":
    projector.stop()
    camera.stop()
    controller.stop()
