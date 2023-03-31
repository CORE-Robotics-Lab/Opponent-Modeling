import numpy as np

def distance(x, y):
    # return np.linalg.norm(x - y)
    # assert x.shape[0] == y.shape[0]
    # assert x.shape[0] == 2
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def proj(a, b):
    """ Project vector a onto b """
    k = (a @ a) / (b @ b)
    return k * b

def clip_theta(theta):
    if theta < -np.pi:
        theta += 2 * np.pi
    elif theta > np.pi:
        theta -= 2 * np.pi
    return theta

def pick_closer_theta(desired_hideout_theta, theta_list):
    min_diff = np.inf
    closest_theta = None
    for theta in theta_list:
        diff = abs(theta - desired_hideout_theta)
        if diff < min_diff:
            min_diff = diff
            closest_theta = theta
    return closest_theta

colors = {
    'blue':   [55,  126, 184],  #377eb8 
    'orange': [255, 127, 0],    #ff7f00
    'green':  [77,  175, 74],   #4daf4a
    'pink':   [247, 129, 191],  #f781bf
    'brown':  [166, 86,  40],   #a65628
    'purple': [152, 78,  163],  #984ea3
    'gray':   [153, 153, 153],  #999999
    'red':    [228, 26,  28],   #e41a1c
    'yellow': [222, 222, 0]     #dede00
}  

opacity = 1
c_str = {k: (v[0]/255,v[1]/255,v[2]/255, opacity)
         for (k, v) in colors.items()}