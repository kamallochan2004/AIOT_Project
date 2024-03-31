import Adafruit_GPIO.SPI as SPI
import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import board
import busio as io
import adafruit_mlx90614
from time import sleep

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

while True:
    targetTemp_c = mlx.object_temperature
    targetTemp_f = celsius_to_fahrenheit(targetTemp_c)
    targetTemp = "{:.2f}".format(targetTemp_f)
    
    # Clear previous text
    draw.rectangle((0, 0, width, height), outline=0, fill=0)
    
    # Write new text
    draw.text((x, y), 'Temperature:{} °F'.format(targetTemp), font=font, fill=255)
    #draw.text((x, y+15), "Temperature fine", font=font, fill=255)

    # Display image.
    disp.image(image)
    disp.display()

    sleep(1)
