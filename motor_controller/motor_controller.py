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

        self.step_range = int(self.degree_range / self.step_angle * self.micro_stepping)

        # initialize GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pins.get_pins(), GPIO.OUT, initial=GPIO.LOW)

        if home_motor:
            self.home_motor()

    def stop(self):
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
            sleep(1 / 500)

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
        self, step: int, freq: int = 225, acceleration_period: float = 1 / 3
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

    @activate_motor
    def increment_step(self):
        if self.step + 1 > self.step_range:
            raise ValueError("step out of range")

        self.dir = Direction.CCW

        GPIO.output(self.pins.STEP, GPIO.HIGH)
        GPIO.output(self.pins.STEP, GPIO.LOW)

        self.step += 1

    @activate_motor
    def decrement_step(self):
        if self.step - 1 < 0:
            raise ValueError("step out of range")

        self.dir = Direction.CW

        GPIO.output(self.pins.STEP, GPIO.HIGH)
        GPIO.output(self.pins.STEP, GPIO.LOW)

        self.step -= 1
