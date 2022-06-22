import pickle
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest, nearest_state, dynamics, dynamics_dt, dynamics_dt_no_motor, HEIGHT_MAX, VEL_MAX, DT, TOTAL_BURN_TIME, VEL_MIN, HEIGHT_MIN
from ValueIteration import weighted_evaluate
import numpy as np

def extract_keys(policy):
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
    return (np.array(times), np.array(vels), np.array(heights))

def plot_policy(policy, threshold=1, result_label=""):
    times, vels, heights = extract_keys(policy)
    data = np.ndarray([len(times), len(heights), len(vels)])

    for i, t in enumerate(times):
        for j, v in enumerate(vels):
            for k, h in enumerate(heights):
                data[i, k, j] = policy[(t, v, h)]

    fig = plt.figure(1, figsize=(6,6))
    main_ax = fig.add_axes([0.1,0.2,0.8,0.7])
    slider_ax  = fig.add_axes([0.32,0.07,0.5,0.05])
    im = main_ax.imshow(scipy.ndimage.gaussian_filter(data[-1,:,:], 1), origin='lower', aspect='auto', vmin=0, vmax=100, cmap='inferno')
    im.set_extent([VEL_MIN, VEL_MAX, HEIGHT_MIN, HEIGHT_MAX])
    main_ax.set_xlabel("Speed towards ground (m/s): ")
    main_ax.set_ylabel("Height (m)")
    my_slider = Slider(slider_ax, 'Burn time remaining (s)', valmin = 0, valmax = max(times), valinit = max(times), valstep=DT)
    colorbar = fig.colorbar(im, ax=main_ax)
    colorbar.set_label(result_label, rotation=270)

    def update(val):
        time_index = int(np.where(times == nearest(times,val))[0])
        new_im = data[time_index,:,:]
        # new_im = scipy.ndimage.gaussian_filter(new_im, 1)
        im.set_data(new_im/threshold)
        colorbar.update_normal(im)
        colorbar.draw_all()
        plt.draw()
    
    my_slider.on_changed(update)
    plt.show()

def sim(policy, apogee):
    np.seterr('raise')
    SIM_DT = 1/100
    time_options, vel_options, height_options = extract_keys(policy)
    vels = []
    heights = []
    actions = []

    state = (TOTAL_BURN_TIME, 0, apogee)
    t, v, h = state

    # This ignition policy was chosen by inspecting the policy cost-to-go at the motor ignition point
    while h*12.5/22 > v:
        # free fallin'
        state = dynamics_dt_no_motor(state, SIM_DT)
        t, v, h = state
        vels.append(v)
        heights.append(h)
        actions.append(0)

    action = 100
    t_next_action = float('inf')
    while t >= 0:
        if t <= t_next_action:
            action = weighted_evaluate(policy, time_options, vel_options, height_options, (nearest(time_options, t), v, h))
            t_next_action = nearest(time_options, t-DT)
        state = dynamics_dt(state, action, SIM_DT)

        t, v, h = state
        vels.append(v)
        heights.append(h)
        actions.append(action/10)
    
    num_steps = len(heights)
    sim_times = np.linspace(0, num_steps*SIM_DT, num_steps)
    plt.plot(sim_times, heights, sim_times, vels, sim_times, actions)
    plt.legend(labels=['height', 'speed', 'throttle/10'])
    plt.show()


# Plot thrust curve
if __name__ == "__main__":
    with open("policy.p", "rb") as f:
        policy = pickle.load(f)
    plot_policy(policy)