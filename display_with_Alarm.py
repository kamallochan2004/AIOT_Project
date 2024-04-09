import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import board
import busio as io
import adafruit_mlx90614
from time import sleep
import RPi.GPIO as GPIO

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32


# Initialize I2C
i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

# 128x32 display with hardware I2C:
disp = Adafruit_SSD1306.SSD1306_128_32(rst=None)
# Initialize library.
disp.begin()

# Clear display.
disp.clear()
disp.display()

# Create blank image for drawing.
# Make sure to create image with mode '1' for 1-bit color.
width = disp.width
height = disp.height
image = Image.new('1', (width, height))
draw = ImageDraw.Draw(image)

padding = 1
y = padding
bottom = height-padding
# Move left to right keeping track of the current x position for drawing shapes.
x = padding

# Load default font.
font = ImageFont.load_default()

buzzPin = 4
led_norm=27
led_alert=22
pwmPin = 26

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzPin, GPIO.OUT)
GPIO.setup(led_norm, GPIO.OUT)
GPIO.setup(led_alert, GPIO.OUT)
GPIO.setup(pwmPin, GPIO.OUT)
buzz = GPIO.PWM(buzzPin, 500)  # Initial frequency



pwm = GPIO.PWM(pwmPin, 50)
pwm.start(0)

angles = [1, 2, 3, 4, 5, 6, 7, 8, 9]

while True:
    targetTemp_c = mlx.object_temperature
    targetTemp_f = celsius_to_fahrenheit(targetTemp_c)
    targetTemp = "{:.2f}".format(targetTemp_f)
    
    # Clear previous text
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    
    # Write new text
    draw.text((x, y), 'Temperature:{} °F'.format(targetTemp), font=font, fill=255)
    GPIO.output(led_norm,1)
    GPIO.output(led_alert,0)
    if float(targetTemp) > 93:
        GPIO.output(led_norm,0)
        GPIO.output(led_alert,1)
        draw.text((x, y+10), "Gate close", font=font, fill=255)
        draw.text((x, y+20), "Access-Denied!!!!", font=font, fill=255)
        buzz.start(80)
        for i in [750, 1000, 1400, 1800]:
            buzz.ChangeFrequency(i)
            sleep(0.15)
        sleep(0.025)
        buzz.stop()
    if float(targetTemp)<93:
        for angle in angles:
            pwmPercent = angle
            pwm.ChangeDutyCycle(pwmPercent)
            sleep(0.1)
        draw.text((x, y+10), "Normal Temperature", font=font, fill=255)
        draw.text((x, y+20), "Gate Open", font=font, fill=255)
    # Display image.
    disp.image(image)
    disp.display()

    sleep(1)

GPIO.cleanup()
