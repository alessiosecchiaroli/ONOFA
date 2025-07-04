import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def get_magnitude(u, v):
    
    sum = 0.0
    counter = 0.0

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            counter += 1
            dy = v[i,j] 
            dx = u[i,j] 
            magnitude = (dx**2 + dy**2)**0.5
            sum += magnitude

    mag_avg = sum / counter

    return mag_avg


def draw_quiver(u,v):
    scale = 3
    ax = plt.figure().gca()
    # ax.imshow(beforeImg, cmap = 'gray')

    magnitudeAvg = get_magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1],8):
            dy = v[i,j] * scale
            dx = u[i,j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            #draw only significant changes
            if magnitude > magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'red')
            elif magnitude < magnitudeAvg:
                ax.arrow(j,i, dx, dy, color = 'lime')
            elif magnitude <=  1e-3:
                break


    plt.draw()
    plt.show()


def plot_midplane(u,label):
    

    y_span = u.shape[0]
    midplane = round(y_span/2)
    u_smoot = savgol_filter(u[midplane,:], 51, 3, axis=0)
    # u_smoot = u[midplane,:]

    plt.plot(u_smoot ,label=label)
    # plt.show()


