import argparse
import sys

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt
from numpy.random import choice

def sampling(buffer, size):
    sample = []
    idx = choice(len(buffer), size)
    print("idx = " , idx)
    for i in idx:
        sample.append(buffer[i])
    return sample

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    rewards = []
    episodes = []
    buffer = []
    buffer_max = 100000

    for i in range(episode_count):
        c_reward = 0
        state_base = env.reset()

        while True:
            action = agent.act(state_base, reward, done)
            state, reward, done, _ = env.step(action)
            buffer.append((state_base, action, state, reward, done))
            if len(buffer) > buffer_max :
                buffer.pop()
            state_base = state
            c_reward += reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        rewards.append(c_reward)
        episodes.append(i)
    print(sampling(buffer, 3))
    plt.plot(rewards)
    plt.show()
    # Close the env and write monitor result info to disk
    env.close()