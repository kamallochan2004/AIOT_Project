import time
import RPi.GPIO as GPIO

buzzPin = 26

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzPin, GPIO.OUT)
buzz = GPIO.PWM(buzzPin, 500)  # Initial frequency
buzz.start(80)  # Increased duty cycle (80%)

try:
    while True:
        # Example alarm pattern (adjust or replace with your desired sound)
        for i in [750, 1000, 1400, 1800]:
            buzz.ChangeFrequency(i)
            time.sleep(0.15)
        time.sleep(0.25)

except KeyboardInterrupt:
    GPIO.cleanup()
    print('GPIO Good to Go')
