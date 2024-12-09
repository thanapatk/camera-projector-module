import Jetson.GPIO as GPIO
import time
from typing import List
from enum import Enum

from utils.singleton import singleton


def sleep(duration):
    start_time = time.perf_counter()
    while True:
        elapsed_time = time.perf_counter() - start_time
        remaining_time = duration - elapsed_time
        if remaining_time <= 0:
            break
        if remaining_time > 0.02:  # Sleep for 5ms if remaining time is greater
            time.sleep(
                max(remaining_time / 2, 0.0001)
            )  # Sleep for the remaining time or minimum sleep interval
        else:
            pass


class StepperPins:
    __slots__ = ["STEP", "DIR", "EN"]

    def __init__(self, step_pin: int, dir_pin: int, enabled_pin: int):
        self.STEP = step_pin
        self.DIR = dir_pin
        self.EN = enabled_pin

    def get_pins(self) -> List[int]:
        return [self.STEP, self.DIR, self.EN]


class Direction(Enum):
    CW = GPIO.LOW
    CCW = GPIO.HIGH


def activate_motor(func):
    def wrapper(self, *args, **kargs):
        self.is_active = True
        result = func(self, *args, **kargs)
        self.is_active = False
        return result

    return wrapper


def activate_motor_no_deactivate(func):
    def wrapper(self, *args, **kargs):
        self.is_active = True
        result = func(self, *args, **kargs)
        return result

    return wrapper


@singleton
class StepperController:
    def __init__(
        self,
        pins: StepperPins,
        step_angle: float,
        degree_range: float,
        micro_stepping: int,
        home_motor: bool = True,
    ):
        self.pins = pins
        self.step_angle = step_angle
        self.micro_stepping = micro_stepping
        self.degree_range = degree_range

        self.__one_degree_step = round(self.micro_stepping / self.step_angle)

        self.step_range = int(self.degree_range / self.step_angle * self.micro_stepping)

        # initialize GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pins.get_pins(), GPIO.OUT, initial=GPIO.LOW)

        if home_motor:
            self.home_motor()

    def stop(self):
        GPIO.output(self.pins.EN, 1)
        GPIO.cleanup()

    @property
    def dir(self) -> Direction:
        return self.__direction

    @dir.setter
    def dir(self, value: Direction):
        self.__direction = value
        GPIO.output(self.pins.DIR, value.value)

    @property
    def is_active(self) -> bool:
        return self.__is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self.__is_active = value
        GPIO.output(self.pins.EN, not value)

    @activate_motor
    def home_motor(self):
        self.dir = Direction.CW
        for _ in range(round(self.step_range * 1.25)):
            GPIO.output(self.pins.STEP, GPIO.HIGH)
            GPIO.output(self.pins.STEP, GPIO.LOW)
            sleep(1 / 550)

        self.is_active = False
        sleep(1)
        self.step = 0

    @activate_motor
    def move_to_step_no_a(self, step: int, freq: int = 2000):
        if not (0 <= step <= self.step_range):
            raise ValueError("step out of range")
        if step == self.step:
            return

        self.dir = Direction.CCW if step - self.step > 0 else Direction.CW

        t = 1 / freq
        for _ in range(abs(step - self.step)):
            GPIO.output(self.pins.STEP, GPIO.HIGH)
            GPIO.output(self.pins.STEP, GPIO.LOW)
            sleep(t)

        self.step = step

    @activate_motor
    def move_to_step(
        self, step: int, freq: int = 550, acceleration_period: float = 1 / 3
    ):
        if not (0 <= step <= self.step_range):
            raise ValueError("step out of range")
        if step == self.step:
            return

        self.dir = Direction.CCW if step - self.step > 0 else Direction.CW

        total_steps = abs(step - self.step)

        a = acceleration_period**-2 * (freq - 0.05 * freq) / total_steps**2

        for n in range(total_steps):
            if n < total_steps * acceleration_period:
                f = -a * (n - total_steps * acceleration_period) ** 2 + freq
            elif n > total_steps * (1 - acceleration_period):
                f = -a * (n - total_steps * (1 - acceleration_period)) ** 2 + freq
            else:
                f = freq

            GPIO.output(self.pins.STEP, GPIO.HIGH)
            GPIO.output(self.pins.STEP, GPIO.LOW)
            sleep(1 / f)

        self.step = step

    @activate_motor_no_deactivate
    def increment_degree(self, degree: int = 1, freq: int = 550):
        self.move_to_step_no_a(self.step + degree * self.__one_degree_step, freq)

    @activate_motor_no_deactivate
    def decrement_degree(self, degree: int = 1, freq: int = 550):
        self.move_to_step_no_a(
            max(0, self.step - degree * self.__one_degree_step), freq
        )

    @activate_motor_no_deactivate
    def increment_step(self, step: int = 1, freq: int = 550):
        if self.step + step > self.step_range:
            raise ValueError("step out of range")

        self.dir = Direction.CCW

        for _ in range(step):
            GPIO.output(self.pins.STEP, GPIO.HIGH)
            GPIO.output(self.pins.STEP, GPIO.LOW)
            sleep(1 / freq)

        self.step += step

    @activate_motor_no_deactivate
    def decrement_step(self, step: int = 1, freq: int = 550):
        if self.step - step < 0:
            raise ValueError("step out of range")

        self.dir = Direction.CW

        for _ in range(step):
            GPIO.output(self.pins.STEP, GPIO.HIGH)
            GPIO.output(self.pins.STEP, GPIO.LOW)
            sleep(1 / freq)

        self.step -= step
