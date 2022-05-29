import pickle
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest, nearest_state, dynamics, dynamics_dt, HEIGHT_MAX, VEL_MAX, DT

def plot_policy(policy, threshold=None):
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

    if threshold:
        data = (data < threshold)*100

    fig = plt.figure(1, figsize=(6,6))
    main_ax = fig.add_axes([0.1,0.2,0.8,0.7])
    slider_ax  = fig.add_axes([0.1,0.1,0.8,0.05])
    im = main_ax.imshow(scipy.ndimage.gaussian_filter(data[0,:,:], 1), origin='lower', aspect='auto', vmin=0, vmax=100)
    im.set_extent([0, VEL_MAX, 0, HEIGHT_MAX])
    main_ax.set_xlabel("Speed towards ground (m/s)")
    main_ax.set_ylabel("Height (m)")
    my_slider = Slider(slider_ax, 'Burn time remaining (s)', valmin = 0, valmax = max(times), valinit = 0)
    colorbar = fig.colorbar(im, ax=main_ax)

    def update(val):
        time_index = times.index(nearest(times, val))
        new_im = data[time_index,:,:]
        # new_im = scipy.ndimage.gaussian_filter(new_im, 1)
        im.set_data(new_im)
        colorbar.update_normal(im)
        colorbar.draw_all()
        plt.draw()
    
    my_slider.on_changed(update)
    plt.show()

def sim(policy, state):
    SIM_DT = 1/100
    times = []
    vels = []
    actions = []
    heights = []

    t, v, h = state
    while t >= 0:
        times.append(t)
        vels.append(v)
        heights.append(h)
        action = policy[nearest_state(*state)]
        actions.append(action/5)
        # state = dynamics(state, action)
        state = dynamics_dt(state, action, SIM_DT)
        t, v, h = state
    
    plt.plot(times, heights, times, vels, times, actions)
    plt.legend(labels=['height', 'speed', 'action/20'])
    plt.gca().invert_xaxis()
    plt.show()


# Plot thrust curve
if __name__ == "__main__":
    with open("policy.p", "rb") as f:
        policy = pickle.load(f)
    plot_policy(policy)