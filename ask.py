import numpy as np

points = np.load("220C.npy")


points_correct1 = points[-2,-2]
points_correct2 = points[-1,-1]

points_real = points[:-3,:-3]

points_real = np.append(points_real,points_correct2)
points_real = np.append(points_real,points_correct1)

np.save("220C_correct",points_real)