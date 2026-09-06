# Source - https://stackoverflow.com/a/15147009
# Posted by Moshe, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-06, License - CC BY-SA 4.0

import cv2

method = cv2.TM_SQDIFF_NORMED

# Read the images from the file
large_image = cv2.imread('x/fingerprint1.png')

height, width = large_image.shape[:2]
large_image = large_image[:, :width // 2]

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("output", 1280, 720)

for i in range(1,5):
    result = cv2.matchTemplate(large_image, small_image, method)
    small_image = cv2.imread(f"x/fingerprint1_{str(i)}.png")

    # We want the minimum squared difference
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows,tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output',large_image)

# The image is only displayed if we call this


cv2.waitKey(0)
