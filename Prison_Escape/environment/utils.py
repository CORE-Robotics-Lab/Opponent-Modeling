import numpy as np
import cv2

def distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x - y)

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

def create_camera_net(start_loc, dist_x, dist_y, spacing, include_camera_at_start=True, board_size=(2428, 2428)):
    """ This function returns a list of camera locations in a square surrounding the prisoner location 
    If the prisoner is at the edge of the board, the cameras will be placed along the board
    This way the number of cameras should be static no matter where the fugitive is
    
    """
    assert dist_x > 0 and dist_y > 0
    assert spacing > 0
    assert board_size[0] > 0 and board_size[1] > 0
    assert start_loc[0] >= 0 and start_loc[1] >= 0
    assert start_loc[0] < board_size[0] and start_loc[1] < board_size[1]
    
    min_x_left = dist_x // 2
    max_x_right = board_size[0] - dist_x // 2
    min_y_bottom = dist_y // 2
    max_y_top = board_size[1] - dist_y // 2

    loc = np.array(start_loc)

    if loc[0] < min_x_left:
        loc[0] = min_x_left
    elif loc[0] > max_x_right:
        loc[0] = max_x_right

    if loc[1] < min_y_bottom:
        loc[1] = min_y_bottom
    elif loc[1] > max_y_top:
        loc[1] = max_y_top

    x_left = loc[0] - dist_x // 2
    x_right = loc[0] + dist_x // 2
    
    y_bot = loc[1] - dist_y // 2
    y_top = loc[1] + dist_y // 2

    x_indices = np.arange(x_left, x_right + spacing, spacing)
    y_indices = np.arange(y_bot, y_top + spacing, spacing)

    # add dimension on numpy array
    bottom_row = np.stack([x_indices, np.ones(len(x_indices)) * y_bot], axis=1)
    top_row = np.stack([x_indices, np.ones(len(x_indices)) * y_top], axis=1)
    left_column = np.stack([np.ones(len(y_indices)) * x_left, y_indices], axis=1)
    right_column = np.stack([np.ones(len(y_indices)) * x_right, y_indices], axis=1)

    camera_locations = np.concatenate([bottom_row, top_row, left_column, right_column], axis=0)

    if include_camera_at_start:
        camera_locations = np.vstack((camera_locations, np.array(start_loc)))
    return camera_locations.astype(int)

if __name__ == "__main__":
    res = create_camera_net((500, 500), dist_x=360, dist_y=360, spacing=30)

    # numpy to list
    print(len(res))   
    print(res) 

    # plot the camera locations
    import matplotlib.pyplot as plt
    plt.scatter(res[:, 0], res[:, 1])
    plt.savefig('test.png')