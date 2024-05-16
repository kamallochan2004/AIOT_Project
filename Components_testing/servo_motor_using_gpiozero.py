from gpiozero import Servo
from time import sleep

servo = Servo(18)

try:
    while True:
        servo.min()
        print("Servo moved to  0 degrees")
        sleep(2)
        
        servo.max()
        print("Servo moved to  180 degrees")
        sleep(1)
        
except KeyboardInterrupt:
    print("Keyboard interrupt detected.Exiting program")
finally:
    servo.close()