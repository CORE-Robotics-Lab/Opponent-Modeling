import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import save_video
import matplotlib.pyplot as plt

def generate_policy_heatmap_video(self, current_state, policy, num_timesteps=2520, num_rollouts=20, end=False, path='simulator/temp.mp4'):
    """
    Generates the heatmap displaying probabilities of ending up in certain cells
    :param current_state: current location of prisoner, current state of world
    :param policy: must input state, output action
    :param num_timesteps: how far in time ahead, remember time is in 15 minute intervals.
    """
    time_between_frames = 60
    num_frames = num_timesteps//time_between_frames
    # Create 3D matrix
    display_matrix = np.zeros((num_frames, self.dim_x + 1, self.dim_y + 1))

    for num_traj in tqdm(range(num_rollouts), desc="generating_heatmap"):
        observation, observation_gt = self.reset()
        frame_index = 0
        for j in range(num_timesteps):
            action = policy.predict(observation_gt, deterministic=False)[0]
            observation, observation_gt, reward, done, _ = self.step(action)
            # update count
            if frame_index > num_frames:
                break
            elif j % time_between_frames == 0:
                display_matrix[frame_index, self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                frame_index += 1
            if done:
                break
        if done:
            for frame_i in range(frame_index, num_frames):
                display_matrix[frame_i, self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
            # self.render('human', show=True)

    imgs = []
    smoothed = []
    for frame_i in tqdm(range(num_frames)):
        matrix = display_matrix[frame_i]
        fig, ax = plt.subplots()
        matrix = np.transpose(matrix)
        # smooth the matrix
        smoothed_matrix = gaussian_filter(matrix, sigma=50)
        smoothed.append(smoothed_matrix)
        # Set 0s to None as they will be ignored when plotting
        # smoothed_matrix[smoothed_matrix == 0] = None
        matrix[matrix == 0] = None
        # Plot the data
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    sharex=False, sharey=True,
                                    figsize=(5, 5))
        # ax1.matshow(display_matrix, cmap='hot')
        # ax1.set_title("Original matrix")
        im = ax1.matshow(smoothed_matrix)
        
        num_hours = str((frame_i * time_between_frames/60).__round__(2))

        ax1.set_title("Heatmap at Time t="+ num_hours + ' hours')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([])
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        cbar.ax.invert_yaxis()
        # plt.show()

        # plt.savefig("simulator/temp" + str(frame_i) + ".png")
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs.append(img)
        plt.close()

    save_video(imgs, path, fps=2)
    return smoothed