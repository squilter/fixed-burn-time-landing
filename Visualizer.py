import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest

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

    data = np.ndarray([len(times), len(vels), len(heights)])

    for i, t in enumerate(times):
        for j, v in enumerate(vels):
            for k, h in enumerate(heights):
                data[i, j, k] = policy[(t, v, h)]

    fig = plt.figure(1, figsize=(6,6))
    main_ax = fig.add_axes([0.1,0.2,0.8,0.7])
    slider_ax  = fig.add_axes([0.1,0.1,0.8,0.05])

    main_ax.imshow(data[0,:,:], aspect='auto')

    my_slider = Slider(slider_ax, 'Burn time remaining (s)', valmin = 0, valmax = len(times), valinit = 0)

    def update(val):
        main_ax.imshow(data[int(val),:,:], aspect='auto')
        main_ax.set_xlabel("Speed towards ground (m/s)")
        main_ax.set_ylabel("Height (m)")
        plt.draw()

    my_slider.on_changed(update)
    plt.show()
