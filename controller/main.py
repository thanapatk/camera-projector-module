import configparser
from motor_controller import StepperController, StepperPins

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
    pins=StepperPins(step_pin=33, dir_pin=35, enabled_pin=36),
)

freq = 1000 * 2  # freq to move to limit 225

if __name__ == "__main__":
    for i in range(10):
        controller.move_to_step(controller.step_range * 3 // 4, freq)
        controller.move_to_step(controller.step_range // 4, freq)

    controller.move_to_step(controller.step_range, 225 * 2)
