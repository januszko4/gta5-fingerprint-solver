import cv2

method = cv2.TM_SQDIFF_NORMED

# Load big image
original = cv2.imread("x/fingerprint1.png")

# Keep only left half
height, width = original.shape[:2]
original = original[:, :width // 2]

# Copy used for drawing
display = original.copy()

cv2.namedWindow("output", cv2.WINDOW_NORMAL)

for i in range(1, 5):

    small_image = cv2.imread(f"x/fingerprint1_{i}.png")

    best_score = float("inf")
    best_location = None
    best_template = None

    # Try different sizes
    for scale in [1]:

        resized = cv2.resize(
            small_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        # Skip if template is bigger than screenshot
        if (resized.shape[0] > original.shape[0] or
                resized.shape[1] > original.shape[1]):
            continue

        result = cv2.matchTemplate(
            original,
            resized,
            method
        )

        mn, _, mnLoc, _ = cv2.minMaxLoc(result)

        # SQDIFF: lower is better
        if mn < best_score:
            best_score = mn
            best_location = mnLoc
            best_template = resized

    # Best match position
    MPx, MPy = best_location

    # Actual size of the scaled template
    trows, tcols = best_template.shape[:2]

    confidence = 1 - best_score

    print(
        f"Fingerprint {i}: "
        f"Position: ({MPx}, {MPy}) "
        f"Confidence: {confidence:.4f} "
        f"Size: {tcols}x{trows}"
    )

    # Draw rectangle around best match
    cv2.rectangle(
        display,
        (MPx, MPy),
        (MPx + tcols, MPy + trows),
        (0, 0, 255),
        2
    )

cv2.imshow("output", display)

cv2.waitKey(0)
cv2.destroyAllWindows()