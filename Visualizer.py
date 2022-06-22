import pickle
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Dynamics import nearest, nearest_state, dynamics, dynamics_dt, dynamics_dt_no_motor, HEIGHT_MAX, VEL_MAX, DT, TOTAL_BURN_TIME, VEL_MIN, HEIGHT_MIN
from ValueIteration import weighted_evaluate
import numpy as np

SIM_DT = 1/100

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

def plot_policy(policy, cost):
    times, vels, heights = extract_keys(policy)
    data_policy = np.ndarray([len(times), len(heights), len(vels)])
    data_cost = np.ndarray([len(times), len(heights), len(vels)])

    for i, t in enumerate(times):
        for j, v in enumerate(vels):
            for k, h in enumerate(heights):
                data_policy[i, k, j] = policy[(t, v, h)]
                data_cost[i, k, j] = cost[(t, v, h)]

    fig = plt.figure(1, figsize=(18,6))
    action_ax = fig.add_axes([0.05,0.2,0.28,0.7])
    cost_ax = fig.add_axes([0.37,0.2,0.28,0.7])
    sim_ax = fig.add_axes([0.69,0.2,0.28,0.7])
    slider_ax  = fig.add_axes([0.2,0.07,0.3,0.05])
    im = action_ax.imshow(scipy.ndimage.gaussian_filter(data_policy[-1,:,:], 1), origin='lower', aspect='auto', vmin=0, vmax=100, cmap='inferno')
    im2 = cost_ax.imshow(scipy.ndimage.gaussian_filter(data_cost[-1,:,:], 1), origin='lower', aspect='auto', vmin=0, vmax=100, cmap='RdYlGn_r')
    im.set_extent([VEL_MIN, VEL_MAX, HEIGHT_MIN, HEIGHT_MAX])
    im2.set_extent([VEL_MIN, VEL_MAX, HEIGHT_MIN, HEIGHT_MAX])
    action_ax.set_xlabel("Speed towards ground (m/s): ")
    action_ax.set_ylabel("Height (m)")
    cost_ax.set_xlabel("Speed towards ground (m/s): ")
    cost_ax.set_ylabel("Height (m)")
    slider = Slider(slider_ax, 'Burn time remaining (s)', valmin = 0, valmax = max(times), valinit = max(times), valstep=DT)
    colorbar = fig.colorbar(im, ax=action_ax)
    colorbar.set_label("Throttle", rotation=270)

    def update_time_slider(val):
        time_index = int(np.where(times == nearest(times,val))[0])
        im.set_data(data_policy[time_index,:,:])
        im2.set_data(data_cost[time_index,:,:]/5)
        colorbar.update_normal(im)
        colorbar.draw_all()
        plt.draw()
    
    def update_sim(event):
        # TODO add third subplot that shows path to landing
        height = event.ydata
        vel = event.xdata
        time = slider.val
        if abs(time-max(times)) > 0.01:
            sim_ax.clear()
            plt.draw()
            return
        if height is not None and vel is not None:
            (vels, heights, actions) = sim_burn(policy, vel, height)
            num_steps = len(heights)
            sim_times = np.linspace(0, num_steps*SIM_DT, num_steps)
            sim_ax.clear()
            sim_ax.set_ylim(0, 20)
            sim_ax.plot(sim_times, heights, sim_times, vels, sim_times, actions)
            sim_ax.legend(labels=['height', 'speed', 'throttle/10'])
            plt.draw()
    
    fig.canvas.mpl_connect('motion_notify_event', update_sim)

    slider.on_changed(update_time_slider)
    update_time_slider(max(times))
    plt.show()

# TODO create apogee slider
def plot_sim(policy, apogee):
    np.seterr('raise')
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

    (burn_vels, burn_heights, burn_actions) = sim_burn(policy, v, h)
    
    num_steps = len(heights+burn_heights)
    sim_times = np.linspace(0, num_steps*SIM_DT, num_steps)
    plt.plot(sim_times, heights+burn_heights, sim_times, vels+burn_vels, sim_times, actions+burn_actions)
    plt.legend(labels=['height', 'speed', 'throttle/10'])
    plt.show()

def sim_burn(policy, vel, height):
    np.seterr('raise')
    state = (TOTAL_BURN_TIME, vel, height)
    t, v, h = state

    time_options, vel_options, height_options = extract_keys(policy)
    vels = []
    heights = []
    actions = []

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
    
    return (vels, heights, actions)


# Plot thrust curve
if __name__ == "__main__":
    with open("policy.p", "rb") as f:
        policy, costs = pickle.load(f)
    plot_policy(policy, costs)