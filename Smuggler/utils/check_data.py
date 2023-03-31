import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
# from ..behavioral_cloning.bc import reconstruct_policy
from simulator.prisoner_env import PrisonerEnv, PrisonerGoalEnv
from sklearn.neural_network import MLPRegressor
from utils import save_video

GOAL = False
# bc_path = "/home/sean/PrisonerEscape/buffers/avoid.pkl"
# # bc_path = "buffers/buffer_single_hideout.pkl"
bc_path = "/nethome/sye40/PrisonerEscape/buffers/hierarchical_rl_rollouts_ground_truth_normal.pkl"

with open(bc_path, 'rb') as handle:
    buffer = pickle.load(handle)

actions = buffer.actions
observations = buffer.observations

print(actions)

actions = np.squeeze(actions, axis=1)
observations = np.squeeze(observations, axis=1)

# print(actions)

# location = observations[:, 51:53]
# goal_location = observations[:, -2:]
# x = np.random.normal(size=1000)

# plt.hist(actions[:, 1], density=True)  # density=False would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data');
# plt.show()

# ax = plt.axes(projection='3d')
# zdata = actions[:, 1]
# xdata = location[:, 0]
# ydata = location[:, 1]
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
# plt.show()

# dtr = MLPRegressor(hidden_layer_sizes=(64, 64))
dtr = DecisionTreeRegressor(max_depth = 20)
print(dtr)

dtr.fit(observations, actions[:, 1])
score = dtr.score(observations, actions[:, 1])
print("R-squared:", score) 

print(dtr.predict([observations[0]]))

# np.random.seed(4)
env = PrisonerEnv()
env.generate_policy_heatmap(env.reset(), policy=dtr.predict, num_timesteps=500, end=True)

if GOAL:
    env = PrisonerGoalEnv(goal_mode="fixed", spawn_mode='normal')
    env.set_hideout_goal(0)
else:
    env = PrisonerEnv(spawn_mode='normal')

observation = env.reset()
images = [env.render('human', show=False, fast=True)]
done = False

i = 0
while not done:
    theta = dtr.predict([observation])
    # action = np.array([(7.5 - 1)/14, theta[0]], dtype=np.float32)
    action = np.array([7.5, theta[0]], dtype=np.float32)
    print(action)
    observation, reward, done, _ = env.step(action)
    
    # print("iteration", i, "reward:", reward, "observation:", observation[:-31*31], "action", action)
    # images.append(env.render('human', show=False))
    # env.render("BC", show=True, fast=True)

    if done:
        print("Done with Episode")
        break

    i += 1

# save_video(images, "traj_video.mp4", fps=10.0)