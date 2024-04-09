import RPi.GPIO as GPIO
from time import sleep

# Setup GPIO pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
pwmPin = 26
GPIO.setup(pwmPin, GPIO.OUT)

pwm = GPIO.PWM(pwmPin, 50)
pwm.start(0)

angles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

try:
    while True:
        for angle in angles:
            pwmPercent = angle
            pwm.ChangeDutyCycle(pwmPercent)
            sleep(0.1)  # Adjust sleep duration for the desired speed

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Stopped")
