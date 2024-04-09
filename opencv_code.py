import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(14, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)

# Track color (initialize to red)
track_color = (0, 0, 255)

# Callback function for mouse click event
def mouse_callback(event, x, y, flags, param):
    global track_color
    if event == cv2.EVENT_LBUTTONDOWN:
        track_color = frame[y, x]

# Capture video from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space for better color comparison
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define color range for tracking (adjust as needed)
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])
    # Threshold the HSV image to get only desired colors
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        # Get the centroid of the largest contour
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Control GPIO based on centroid position
            if cx < 140:
                GPIO.output(4, GPIO.HIGH)
                GPIO.output(14, GPIO.HIGH)
                GPIO.output(17, GPIO.HIGH)
                GPIO.output(18, GPIO.LOW)
                print("Turn Right")
            elif cx > 200:
                GPIO.output(4, GPIO.HIGH)
                GPIO.output(14, GPIO.LOW)
                GPIO.output(17, GPIO.HIGH)
                GPIO.output(18, GPIO.HIGH)
                print("Turn Left")
            elif cy < 170:
                GPIO.output(4, GPIO.HIGH)
                GPIO.output(14, GPIO.LOW)
                GPIO.output(17, GPIO.HIGH)
                GPIO.output(18, GPIO.LOW)
                print("Go Forward")
            else:
                GPIO.output(4, GPIO.HIGH)
                GPIO.output(14, GPIO.HIGH)
                GPIO.output(17, GPIO.HIGH)
                GPIO.output(18, GPIO.HIGH)

            # Draw a circle around the centroid
            cv2.circle(frame, (cx, cy), 16, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for mouse click to update tracked color
    cv2.setMouseCallback('Frame', mouse_callback)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()