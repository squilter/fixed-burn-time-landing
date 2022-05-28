import pickle
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest, HEIGHT_MAX, VEL_MAX

def plot_policy(policy):
    times = set()
    vels = set()
    heights = set()
    for t, vel, height in policy.keys():
        times.add(t)
        vels.add(vel)
        heights.add(height)
    times = list(times)
    times.sort()
    vels = list(vels)
    vels.sort()
    heights = list(heights)
    heights.sort()

    data = np.ndarray([len(times), len(heights), len(vels)])

    for i, t in enumerate(times):
        for j, v in enumerate(vels):
            for k, h in enumerate(heights):
                data[i, k, j] = policy[(t, v, h)]

    fig = plt.figure(1, figsize=(6,6))
    main_ax = fig.add_axes([0.1,0.2,0.8,0.7])
    slider_ax  = fig.add_axes([0.1,0.1,0.8,0.05])
    im = main_ax.imshow(scipy.ndimage.gaussian_filter(data[0,:,:], 1), origin='lower', aspect='auto')
    im.set_extent([0, VEL_MAX, 0, HEIGHT_MAX])
    main_ax.set_xlabel("Speed towards ground (m/s)")
    main_ax.set_ylabel("Height (m)")
    my_slider = Slider(slider_ax, 'Burn time remaining (s)', valmin = 0, valmax = max(times), valinit = 0)
    fig.colorbar(im, ax=main_ax)

    def update(val):
        time_index = times.index(nearest(times, val))
        im.set_data(scipy.ndimage.gaussian_filter(data[time_index,:,:], 1))
        plt.draw()
    
    my_slider.on_changed(update)
    plt.show()

# Plot thrust curve
if __name__ == "__main__":
    with open("policy.p", "rb") as f:
        policy = pickle.load(f)
    plot_policy(policy)