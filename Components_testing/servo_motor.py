import RPi.GPIO as GPIO
from time import sleep

# Setup GPIO pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
pwmPin = 18
GPIO.setup(pwmPin, GPIO.OUT)

pwm = GPIO.PWM(pwmPin, 50)
pwm.start(0)

angles = [5,11]

try:
    while True:
        for angle in angles:
            pwmPercent = angle
            pwm.ChangeDutyCycle(pwmPercent)
            sleep(1.5)  # Adjust sleep duration for the desired speed

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Stopped")
