import board
import busio as io
import adafruit_mlx90614

from time import sleep

# Define a function for Fahrenheit conversion
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

while True:
    ambientTemp_c = mlx.ambient_temperature
    targetTemp_c = mlx.object_temperature

    # Convert temperatures to Fahrenheit
    ambientTemp_f = celsius_to_fahrenheit(ambientTemp_c)
    targetTemp_f = celsius_to_fahrenheit(targetTemp_c)

    # Format temperatures to two decimal places
    ambientTemp = "{:.2f}".format(ambientTemp_f)
    targetTemp = "{:.2f}".format(targetTemp_f)

    print("Ambient Temperature:", ambientTemp, "°F")
    print("Target Temperature:", targetTemp, "°F")

    # Delay between readings (adjust as needed)
    sleep(1)  
