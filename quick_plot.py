import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

u = np.load('u-displacement.npy')

u_smoot = savgol_filter(u[340,:], 51, 3, axis=0)

plt.plot(u_smoot)
plt.show()