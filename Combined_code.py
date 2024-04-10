import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import board
import busio as io
import adafruit_mlx90614
from time import sleep
import RPi.GPIO as GPIO
import cv2
from picamera2 import Picamera2
import time


import time
import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont



buzzer_pin = 4
frequencies = [750, 1000, 1400, 1800]
durations = [0.15, 0.15, 0.15, 0.15]
servo_pin = 26
angles = [1, 2, 3, 4, 5, 6, 7]
led_norm=27
led_alert=22


def control_led(led_pin, state):
  GPIO.setwarnings(False)
  GPIO.setmode(GPIO.BCM)  # Only set mode once if needed (avoid redundant calls)
  GPIO.setup(led_pin, GPIO.OUT)

  GPIO.output(led_pin, state)

def control_servo(servo_pin, angles):
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin, GPIO.OUT)

    pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz pulse frequency
    pwm.start(0)  # Initial duty cycle (0%)

    for angle in angles:
            # Map angle (0-10) to duty cycle (roughly 5%-10%)
        pwm_duty_cycle = (angle / 10) * 5 + 5  # Adjust this formula as needed for your servo
        pwm.ChangeDutyCycle(pwm_duty_cycle)
        sleep(0.1)  # Adjust sleep duration for desired speed



def play_alarm(buzzer_pin, frequencies, durations):
   
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer_pin, GPIO.OUT)

    buzz = GPIO.PWM(buzzer_pin, 500)  # Initial frequency
    buzz.start(80)  # Increased duty cycle (80%)

    try:
        for frequency, duration in zip(frequencies, durations):
            buzz.ChangeFrequency(frequency)
            time.sleep(duration)

    except KeyboardInterrupt:
        pass  # No need to print a message here

    finally:
        GPIO.cleanup()


def display_text(text1, text2):
    
    # Display setup (assuming hardware I2C connection)
    disp = Adafruit_SSD1306.SSD1306_128_32(rst=None)
    disp.begin()

    # Clear display
    disp.clear()
    disp.display()

    # Create image for drawing
    width = disp.width
    height = disp.height
    image = Image.new('1', (width, height))
    draw = ImageDraw.Draw(image)

    padding = 1
    y = padding
    bottom = height - padding

    # Load default font
    font = ImageFont.load_default()

    # Improved text placement with centering
    text1_width, _ = draw.textsize(text1, font=font)
    text2_width, _ = draw.textsize(text2, font=font)
    x1 = (width - text1_width) // 2  # Center text1 horizontally
    x2 = (width - text2_width) // 2  # Center text2 horizontally

    draw.text((x1, y), text1, font=font, fill=255)
    draw.text((x2, y + 15), text2, font=font, fill=255)

    # Display image
    disp.image(image)
    disp.display()


def read_target_temperature():
    i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
    mlx = adafruit_mlx90614.MLX90614(i2c)

    targetTemp_c = mlx.object_temperature

    targetTemp_f = (targetTemp_c * 9 / 5) + 32
    targetTemp = "{:.2f}".format(targetTemp_f)

    return targetTemp

# Example usage
display_text("GROUP-2", "AIOT-Project")

def detect_face(dispW, dispH):
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (dispW, dispH)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.controls.FrameRate = 30
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    fps = 0
    pos = (30, 60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    height = 1.5
    weight = 3
    myColor = (0, 0, 255)

    faceCascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

    while True:
        tStart = time.time()
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frameGray, 1.3, 5)

        face_detected = len(faces) > 0

        # Draw rectangle around each detected face
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), myColor, 3)

        # Calculate FPS
        tEnd = time.time()
        loopTime = tEnd - tStart
        fps = 0.9 * fps + 0.1 * (1 / loopTime)
        cv2.putText(frame, str(int(fps)) + ' FPS', pos, font, height, myColor, weight)

        #Temperature 
        targetTemp = str(read_target_temperature())
        
        # Print face detection status based on current frame
        if face_detected:
            print("Face detected!")
            display_text("Temperature ",targetTemp)
            if float(targetTemp)>92:
                play_alarm(buzzer_pin, frequencies, durations)
                control_led(led_norm, False)  # Turn on normal LED
                control_led(led_alert, True)  # Turn off alert LED
            else:
                control_led(led_norm, True)  # Turn on normal LED
                control_led(led_alert, False)  # Turn off alert LED
                control_servo(servo_pin, angles)
        else:
            print("Face not detected.")
            display_text("No Face","Scan Again")
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    picam2.stop()
    cv2.destroyAllWindows()
    return face_detected, frame  # Return face detection result and frame (optional)


# Main function
if __name__=="__main__":
    dispW,dispH=720,480
    
    # Initialize display
    display_text("GROUP-2", "AIOT-Project")
   
    face_detected, frame = detect_face(dispW, dispH)
        
