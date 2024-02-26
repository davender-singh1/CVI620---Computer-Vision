import cv2
import numpy as np

# Part 1a: Open the image and convert it to grayscale
image = cv2.imread('credit.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part 1b: Use Sobel filter to find edges
# Sobel filter for dx = 1, dy = 0
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)

# Sobel filter for dx = 0, dy = 1
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Sobel filter for dx = 1, dy = 1
sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1)
sobel_xy_abs = cv2.convertScaleAbs(sobel_xy)

# Display results for Sobel filters
cv2.imshow('Sobel dx=1, dy=0 (CV_64F)', sobel_x_abs)
cv2.imshow('Sobel dx=0, dy=1 (CV_64F)', sobel_y_abs)
cv2.imshow('Sobel dx=1, dy=1 (CV_64F)', sobel_xy_abs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert Sobel results to CV_8U
sobel_x_8u = cv2.convertScaleAbs(sobel_x)
sobel_y_8u = cv2.convertScaleAbs(sobel_y)
sobel_xy_8u = cv2.convertScaleAbs(sobel_xy)

# Display Sobel results with CV_8U datatype
cv2.imshow('Sobel dx=1, dy=0 (CV_8U)', sobel_x_8u)
cv2.imshow('Sobel dx=0, dy=1 (CV_8U)', sobel_y_8u)
cv2.imshow('Sobel dx=1, dy=1 (CV_8U)', sobel_xy_8u)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part 1c: Apply Canny edge detector with different thresholds
canny_low_threshold = 50
canny_high_threshold = 150

edges_low = cv2.Canny(gray_image, canny_low_threshold, canny_high_threshold)
edges_high = cv2.Canny(gray_image, canny_low_threshold * 2, canny_high_threshold * 2)

# Display Canny edge detection results
cv2.imshow('Canny Edges (Low Threshold)', edges_low)
cv2.imshow('Canny Edges (High Threshold)', edges_high)
cv2.waitKey(0)
cv2.destroyAllWindows()
