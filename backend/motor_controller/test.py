import Jetson.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

STEP = 33
DIR = 35
EN = 36

freq = 2000
sleep_time = 1 / (2 * freq)

GPIO.setup([STEP, DIR, EN], GPIO.OUT, initial=GPIO.LOW)

# GPIO.output(EN, GPIO.HIGH)

"""
ratio 20:68

output 80 deg -> input 80 * 68 / 20
"""

full_range_deg = 8 * 80 * 68 / 20

try:
    while True:
        GPIO.output(DIR, GPIO.LOW)
        for _ in range(int(full_range_deg / 1.8)):
            GPIO.output(STEP, GPIO.HIGH)
            sleep(sleep_time)  # 50 Hz
            GPIO.output(STEP, GPIO.LOW)
            sleep(sleep_time)  # 50 Hz

        sleep(0.1)

        GPIO.output(DIR, GPIO.HIGH)
        for _ in range(int(full_range_deg / 1.8)):
            GPIO.output(STEP, GPIO.HIGH)
            sleep(sleep_time)  # 50 Hz
            GPIO.output(STEP, GPIO.LOW)
            sleep(sleep_time)  # 50 Hz

        sleep(0.1)
except:
    pass

GPIO.output(EN, GPIO.HIGH)

GPIO.cleanup()
