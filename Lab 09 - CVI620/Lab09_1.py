import cv2
import numpy as np
from datetime import datetime

# Initialize the counter for saved images
save_counter = 0

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read the first frame
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read from webcam")

# Convert the first frame to grayscale and use it as the background
background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read the current frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the difference between the current frame and the background
    difference = cv2.absdiff(background, gray_frame)

    # Apply threshold to the difference
    thresh = cv2.threshold(difference, 128, 255, cv2.THRESH_BINARY)[1]

    # Count the number of white pixels in the threshold image
    white_pixels = np.sum(thresh) / 255
    total_pixels = frame.shape[0] * frame.shape[1]
    percentage = (white_pixels / total_pixels) * 100

    # Check for significant changes in the image
    if percentage > 1.5:  # Change 1.5 to the desired sensitivity
        print(f"Activity detected at: {datetime.now()}")
        # Save the current color frame with timestamp
        save_counter += 1
        filename = f"frame_{save_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(f'background_{save_counter}.jpg', background)
        cv2.imwrite(f'difference_{save_counter}.jpg', difference)

        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        # Update the background image
        background = gray_frame
        # Wait for 5 seconds
        cv2.waitKey(5000)

    # Display the threshold image (difference image)
    cv2.imshow('Threshold Image', thresh)

    # Exit the program by entering esc or 'q'
    key = cv2.waitKey(1)
    if key in [27, ord('q')]:  # 27 is the escape key
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
