import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# def detect_single_circle(image_path, blur_ksize, dp, minDist,
#                          param1, param2, minRadius, maxRadius):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

#     circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
#                                param1=param1, param2=param2,
#                                minRadius=minRadius, maxRadius=maxRadius)

#     if circles is not None and len(circles[0]) == 1:
#         return tuple(np.uint16(np.around(circles[0][0]))), img
#     return None, img

def detect_single_circle(image_path, blur_ksize, dp, minDist,
                               param1, param2, minRadius, maxRadius):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path)
    img_blur = cv2.GaussianBlur(img_gray, (blur_ksize, blur_ksize), 0)

    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(image_path)
        plt.show()
    else:
        print(f"No circles detected in {image_path}")

def calculate_offset(c1, c2):
    x1, y1 = c1[0], c1[1]
    x2, y2 = c2[0], c2[1]
    return x2 - x1, y2 - y1

# Parameter grid
blur_ksizes = [3, 5, 7]
param1_vals = [50, 100, 150]
param2_vals = [20, 30, 40]
radii = [(10, 100), (20, 80), (15, 90)]

results = []

for blur, p1, p2, (minR, maxR) in product(blur_ksizes, param1_vals, param2_vals, radii):
    c1, img1 = detect_single_circle("Exp_pics/220C/220C_ref.bmp", blur, 1.2, 30, p1, p2, minR, maxR)
    c2, img2 = detect_single_circle("Exp_pics/220C/220C_9bar.bmp", blur, 1.2, 30, p1, p2, minR, maxR)

    print(f"Testing blur={blur}, param1={p1}, param2={p2}, minR={minR}, maxR={maxR}")
    print(f"  - ref: {'1 circle' if c1 else 'None or >1'}; 9bar: {'1 circle' if c2 else 'None or >1'}")

    if c1 and c2:
        r1, r2 = c1[2], c2[2]
        print(f"    -> radii: ref={r1}, 9bar={r2}")
        if abs(r1 - r2) <= 5:  # Temporarily allow up to 5 pixels diff
            dx, dy = calculate_offset(c1, c2)
            results.append({
                "blur": blur,
                "param1": p1,
                "param2": p2,
                "minR": minR,
                "maxR": maxR,
                "radius_ref": r1,
                "radius_9bar": r2,
                "dx": dx,
                "dy": dy,
                "circle_ref": c1,
                "circle_9bar": c2
            })


# Show best match
if results:
    best = results[0]
    print("✅ Best Match Found:")
    for k, v in best.items():
        if 'circle' not in k:
            print(f"{k}: {v}")

    # Visualize circles
    def draw_circle(img_path, circle):
        img = cv2.imread(img_path)
        x, y, r = circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        return img

    ref_drawn = draw_circle("Exp_pics/220C/220C_ref.bmp", best["circle_ref"])
    bar_drawn = draw_circle("Exp_pics/220C/220C_9bar.bmp", best["circle_9bar"])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(ref_drawn, cv2.COLOR_BGR2RGB))
    plt.title("Reference")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bar_drawn, cv2.COLOR_BGR2RGB))
    plt.title("9 bar")
    plt.show()
else:
    print("❌ No reliable circle pair found with matching diameters.")

