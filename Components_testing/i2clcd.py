from signal import pause
from rpi_lcd import LCD

lcd = LCD()

try:
    
    lcd.text("AIOT_PROJECT", 1)
    lcd.text("GROUP 20", 2)

    pause()

except KeyboardInterrupt:
    pass

finally:
    lcd.clear()
