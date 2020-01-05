import numpy as np
import gym
import random
import torch
import cv2

class Wrapper(object):
    """
    Wrapper used to simplify interactions with the environment
    """

    def __init__(self):
        self.env = gym.make('Breakout-ramNoFrameskip-v4')
        self.last_lives = 0
        self.last_frames = []

    def reset(self):
        """
        Resets the environment
        Executes a random number of no op actions between 1 and N_NO_OP
        Create the initial state by stacking four identical frames
        """
        self.env.reset()
        self.last_lives = 0
        # Perform N times the no op action
        # Those N iterations are not part of the learning process (counter not incremented, epsilon unchanged, etc.)
        for i in range(random.randint(1, 20)):
            self.env.step(1)
        processed_frame = self.preprocess_frame(self.env.render(mode='rgb_array'))
        # Create the initial state with four identical frames
        self.last_frames = []
        for _ in range(4):
            self.last_frames.append(processed_frame)

    def step(self, action):
        """
        Executes the action in the environment
        Return the pre-processed resulting frame,
        the reward and if the agent lost a life or not
        :param action: Integer, action to perform
        :return: (processed_new_frame, reward, terminal, terminal_life_lost)
        """
        _, reward, done, s = self.env.step(action)

        terminal_life_lost = done
        if s['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        self.last_lives = s['ale.lives']

        processed_new_frame = self.preprocess_frame(self.env.render(mode='rgb_array'))
        self.last_frames.append(processed_new_frame)
        self.last_frames.pop(0)

        return processed_new_frame, reward, done, terminal_life_lost

    def preprocess_frame(self, f):
        """
        Pre-process one frame
        :param f: Raw frame from the environment
        :return: Grayscaled, cropped and rescaled to 84x84 frame
        """
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[34:-18, :]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        return f

    def get_state(self):
        """
        Process the k last frames to an unsqueezed tensor on the selected and return it as the current state
        """
        state = np.array(self.last_frames) / 255.
        return torch.tensor(state, dtype=torch.float64).unsqueeze(0)

    def render(self, mode = 'human'):
        if mode == 'human':
            self.env.render(mode=mode)