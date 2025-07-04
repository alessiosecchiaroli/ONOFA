import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from Pyramidal_Horn_Schunck import HS_pyramidal


# root = os.getcwd()

# video_path = os.path.join(root, './vids/expanded_tile.mp4')

def draw_trajectories(image, trajectories):
    plt.figure()
    plt.imshow(image, cmap='gray')
    for traj in trajectories:
        xs, ys = zip(*traj)
        plt.plot(xs, ys, linewidth=1)
    plt.title("Tracked Motion Trajectories (Horn-Schunck)")
    plt.axis('off')
    plt.show()


def video_OF(video_path):
    cap = cv.VideoCapture(video_path)

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        return

    # Convert the first frame to grayscale
    first_frame_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    prev_gray = first_frame_gray.copy()

    # Initialize grid of points to track
    step = 10
    y, x = np.mgrid[0:first_frame_gray.shape[0]:step, 0:first_frame_gray.shape[1]:step]
    points = np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.float32)  # (N, 2)

    # Create list of trajectories: one list of (x, y) per point
    trajectories = [[tuple(p)] for p in points]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute optical flow
        u, v = HS_pyramidal(prev_gray, frame_gray, 10, 1, 0.1, 5)
        flow = np.stack((u, v), axis=-1)

        new_points = []
        for i, (x, y) in enumerate(points):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < flow.shape[1] and 0 <= yi < flow.shape[0]:
                dx, dy = flow[yi, xi]
                new_x = x + dx
                new_y = y + dy
                new_points.append([new_x, new_y])
                trajectories[i].append((new_x, new_y))
            else:
                new_points.append([x, y])  # or keep previous
        points = np.array(new_points, dtype=np.float32)
        prev_gray = frame_gray.copy()

    cap.release()
    cv.destroyAllWindows()

    # After processing, draw the trajectories
    draw_trajectories(first_frame_gray, trajectories)

    
root = os.getcwd()
video_path = os.path.join(root, './vids/expanded_tile.mp4')
video_OF(video_path)