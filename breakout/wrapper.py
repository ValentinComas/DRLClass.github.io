import numpy as np
import gym
import random
import torch
import cv2

class AtariPreprocessing(object):
    """
    Wrapper used to simplify interactions with the environment
    """

    def __init__(self):
        self.env = gym.make('Breakout-ramNoFrameskip-v4')
        self.env = gym.wrappers.Monitor(self.env, "recording", force=True)
        self.ll = 0
        self.frames = []

    def reset(self):
        self.env.reset()
        self.ll = 0
        for _ in range(random.randint(1, 30)):
            _, _, _, _ = self.env.step(1)
        processed_frame = self.preprocess_frame(self.env.render(mode='rgb_array'))
        self.frames = [None] * 4
        for i in range(4):
            self.frames[i] = processed_frame
        return self.frames

    def step(self, action):
        r = 0.0
        for i in range(4):
            _, reward, terminal, obs = self.env.step(action)
        for i in range(4):
            _, reward, terminal, obs = self.env.step(action)
            terminal_life_lost = terminal
            if obs['ale.lives'] < self.ll:
                terminal_life_lost = True
            self.ll = obs['ale.lives']

            processed_new_frame = self.preprocess_frame(self.env.render(mode='rgb_array'))
            self.frames.append(processed_new_frame)
            self.frames.pop(0)
            r += reward

        return self.frames, r, terminal, terminal_life_lost

    def preprocess_frame(self, f):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[34:-18, :]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        return f
