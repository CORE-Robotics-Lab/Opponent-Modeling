"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""

import yaml
import argparse
import torch
import numpy as np
import random
import os

def save_video(ims, filename, fps=30.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

def get_configs():
    """
    Parse command line arguments and return the resulting args namespace
    """
    parser = argparse.ArgumentParser("Train Filtering Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded, args.config

def evaluate_mean_reward(env, policy, num_iter, show=False):
    mean_return = 0.0

    for _ in range(num_iter):
        # if goal < 0:
        #     env.set_hideout_goal(random.randint(0, 2))
        # env = gym.make("PrisonerEscape-v0")
        observation = env.reset()
        done = False
        episode_return = 0.0
        i = 0
        while not done:
            action = policy(observation)
            observation, reward, done, _ = env.step(action[0])
            episode_return += reward

            if show:
                env.render('Policy', show=True, fast=True)

            if done:
                break

        mean_return += episode_return / num_iter
        print(episode_return)

    return mean_return

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def sample_n_times(pi, mu, sigma, n, device='cpu'):
    """
    Draw n samples from a MoG.
    pi: (B, G)
    mu: (B, G, D)
    sigma: (B, G, D)
    # B Batch
    # n number of samples
    # G number of gaussians
    # D output dimension
    """
    # B = pi.size(0)
    # G = mu.size(1)
    # D = mu.size(2)

    # Choose which gaussian we'll sample from
    pis = torch.multinomial(pi, n, replacement=True) # (B, n)

    def gather_and_select(pis, obj):
        all_samples = []
        for index in range(pi.size(0)):
            pi_indexed = pis[index]
            # print(pi_indexed)
            obj_indexed = obj[index]
            # print(mus)
            samples = torch.index_select(obj_indexed, 0, pi_indexed)
            all_samples.append(samples)
        return torch.stack(all_samples, dim=0)
        
    mean_samples = gather_and_select(pis, mu)
    variance_samples = gather_and_select(pis, sigma)
    gaussian_noise = torch.randn(variance_samples.shape, device=device, requires_grad=False)

    return gaussian_noise * variance_samples + mean_samples

def evaluate_mean_reward(env, policy, num_iter, show=False):
    mean_return = 0.0

    for _ in range(num_iter):
        # if goal < 0:
        #     env.set_hideout_goal(random.randint(0, 2))
        # env = gym.make("PrisonerEscape-v0")
        observation = env.reset()
        done = False
        episode_return = 0.0
        i = 0
        while not done:
            action = policy(observation)
            observation, reward, done, _ = env.step(action[0])
            episode_return += reward

            if show:
                env.render('Policy', show=True, fast=True)

            if done:
                break

        mean_return += episode_return / num_iter
        print(episode_return)

    return mean_return

def save_video(ims, filename, fps=30.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        # b, g, r
        writer.write(im)
    writer.release()