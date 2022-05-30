import pickle
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest, nearest_state, dynamics, dynamics_dt, HEIGHT_MAX, VEL_MAX, DT
from ValueIteration import weighted_evaluate

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
    return (times, vels, heights)

def plot_policy(policy, threshold=None):
    times, vels, heights = extract_keys(policy)
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
    np.seterr('raise')
    SIM_DT = 1/100
    time_options, vel_options, height_options = extract_keys(policy)
    times = []
    vels = []
    heights = []
    actions = []

    t, v, h = state
    while t >= 0:
        times.append(t)
        vels.append(v)
        heights.append(h)

        # TODO choose when to ignite motor

        # option 1: plot the transitions through the planned, discretized state space
        # action = policy[nearest_state(*state)]
        # state = dynamics(state, action)

        # option 2: plot the dynamics at a higher rate
        evaluate_times = (nearest(time_options, t+DT/2), nearest(time_options, t-DT/2))
        eval_weight = 1
        if abs(evaluate_times[0] - evaluate_times[1]) > 0.00001:
            eval_weight = (t-evaluate_times[1])/(evaluate_times[0]-evaluate_times[1])
        assert eval_weight <= 1
        eval_actions = (weighted_evaluate(policy, time_options, vel_options, height_options, (evaluate_times[0], v, h)),
                        weighted_evaluate(policy, time_options, vel_options, height_options, (evaluate_times[1], v, h)))
        action = eval_actions[0]*eval_weight + eval_actions[1]*(1-eval_weight)

        # action = weighted_evaluate(policy, time_options, vel_options, height_options, (nearest(time_options, t-DT), v, h))
        state = dynamics_dt(state, action, SIM_DT)

        actions.append(action/10)
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