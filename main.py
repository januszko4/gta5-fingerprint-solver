import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def best_match_at_scales(original_gray, template_gray, scales):
    best = (float("inf"), None, None, None)  # score, loc, (w,h), scale
    for scale in scales:
        resized = cv2.resize(template_gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        th, tw = resized.shape[:2]
        if th < 5 or tw < 5 or th > original_gray.shape[0] or tw > original_gray.shape[1]:
            continue
        result = cv2.matchTemplate(original_gray, resized, cv2.TM_SQDIFF_NORMED)
        mn, _, mnLoc, _ = cv2.minMaxLoc(result)
        if mn < best[0]:
            best = (mn, mnLoc, (tw, th), scale)
    return best

original = cv2.imread("x/fingerprint1_test.png")
height, width = original.shape[:2]
original = original[:, :width // 2]
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
display = original.copy()

cv2.namedWindow("output", cv2.WINDOW_NORMAL)

templates_gray = {}
for i in range(1, 5):
    small = cv2.imread(f"x/fingerprint1_{i}.png")
    if small is None:
        print(f"Fingerprint {i}: file missing")
        continue
    templates_gray[i] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

# --- Step 1: calibrate scale once, using the first available template ---
calib_i = next(iter(templates_gray))
coarse_scales = np.linspace(0.3, 1.5, 13)
score, loc, size, scale = best_match_at_scales(original_gray, templates_gray[calib_i], coarse_scales)

fine_scales = np.linspace(max(0.05, scale - 0.08), scale + 0.08, 9)
score, loc, size, scale = best_match_at_scales(original_gray, templates_gray[calib_i], fine_scales)

print(f"Calibrated scale: {scale:.4f} (from fingerprint {calib_i}, confidence {1-score:.4f})")

# --- Step 2: reuse that scale for every template, fast single match each ---
def process(i):
    template_gray = templates_gray[i]
    resized = cv2.resize(template_gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    th, tw = resized.shape[:2]
    if th > original_gray.shape[0] or tw > original_gray.shape[1]:
        return None
    result = cv2.matchTemplate(original_gray, resized, cv2.TM_SQDIFF_NORMED)
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)
    return i, mn, mnLoc, (tw, th)

with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(process, [i for i in templates_gray]))

for r in results:
    if r is None:
        continue
    i, score, (MPx, MPy), (tw, th) = r
    confidence = 1 - score
    print(f"Fingerprint {i}: Position: ({MPx}, {MPy}) Confidence: {confidence:.4f} Size: {tw}x{th}")
    cv2.rectangle(display, (MPx, MPy), (MPx + tw, MPy + th), (0, 0, 255), 2)

cv2.imshow("output", display)
cv2.waitKey(0)
cv2.destroyAllWindows()