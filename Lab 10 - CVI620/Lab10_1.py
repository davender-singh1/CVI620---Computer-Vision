
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:/Users/Davender Singh/PycharmProjects/Lab10_1/5_of_diamonds.png'
img = cv2.imread(image_path)

# Step 1: Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step1_gray.png', gray)

# Step 2: Thresholding to get binary image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step2_thresh.png', thresh)

# Step 3: Noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step3_opening.png', opening)

# Step 4: Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step4_sure_bg.png', sure_bg)

# Step 5: Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step5_sure_fg.png', sure_fg)

# Step 6: Finding unknown region
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step6_unknown.png', unknown)

# Step 7: Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
cv2.imwrite('C:/Users/Davender Singh/PycharmProjects/Lab10_1/step7_markers.png', np.uint8(markers * (255 / np.max(markers))))

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply the Watershed algorithm
markers = cv2.watershed(img, markers)

# Creating a black image to draw contours on
black_image = np.zeros_like(img)

# Draw green contours on the black image
black_image[markers == -1] = [0, 255, 0]

# Save and display the result
output_path = 'C:/Users/Davender Singh/PycharmProjects/Lab10_1/5_of_diamonds_segmented.png'
cv2.imwrite(output_path, black_image)

# Display the result using matplotlib
black_image_rgb = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
plt.imshow(black_image_rgb)
plt.axis('off')
plt.show()
