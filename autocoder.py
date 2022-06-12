from string import Template
from Visualizer import extract_keys
import numpy as np

template = Template("""
#include <stdint.h>

#define TIME_BUCKETS $T
#define VEL_BUCKETS $V
#define HEIGHT_BUCKETS $H

uint8_t policy[TIME_BUCKETS][VEL_BUCKETS][HEIGHT_BUCKETS] = {
 $values
};

""")

def write_policy(policy):
    time_options, vel_options, height_options = extract_keys(policy)
    time_buckets = len(time_options)
    vel_buckets = len(vel_options)
    height_buckets = len(height_options)

    arr = np.zeros([time_buckets, vel_buckets, height_buckets], dtype='uint8')
    
    for i, t in enumerate(time_options):
        for j, v in enumerate(vel_options):
            for k, h in enumerate(height_options):
                arr[i][j][k] = int(policy[(t,v,h)])

    v = np.array2string(arr.flatten(order='C'), separator=', ', threshold=float('inf'), edgeitems=float('inf'))[1:-1]

    with open('policy.c', 'w') as f:
        f.write(template.substitute(T=time_buckets, V=vel_buckets, H=height_buckets, values=v))