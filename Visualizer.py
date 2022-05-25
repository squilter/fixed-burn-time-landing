import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

    fig = plt.figure()

    ax1 = fig.add_axes([0, 0, 1, 0.8], projection = '3d')
    ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

    s = Slider(ax = ax2, label = 'Burn time remaining (s)', valmin = min(times), valmax = max(times), valinit = max(times))

    def update(val):
        time = nearest(np.array(list(times)), s.val)
        x,y = np.meshgrid(list(vels), list(heights))
        z = np.ndarray([len(heights), len(vels)])

        print(x.shape)
        print(y.shape)
        print(z.shape)

        for i in range(len(heights)): # 401
            for j in range(len(vels)): # 101
                v = x[i,j]
                h = y[i,j]
                z[i, j] = policy[(time, v, h)]
        
        ax1.cla()
        ax1.plot_surface(x, y, z, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
        ax1.set_zlim(0, 100)

        ax1.set_xlabel("Speed towards ground (m/s)")
        ax1.set_ylabel("Height (m)")

    s.on_changed(update)
    update(0.0)

    plt.show()

