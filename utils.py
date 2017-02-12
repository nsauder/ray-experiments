from collections import deque
from contextlib import contextmanager
import gym
import numpy as np
import numpy.random as rng
import random
import scipy
import time
import toolz
from toolz import dicttoolz


def chunk_maps(datamaps, factory=dict):
    if isinstance(datamaps, dict):
        return datamaps

    keys = datamaps[0].keys()
    datamap = {k: list(toolz.pluck(k, datamaps)) for k in keys}

    return dicttoolz.valmap(np.array,
                            datamap,
                            factory=factory)


def get_shape(gym_space):
    # HACK
    if type(gym_space) == gym.spaces.discrete.Discrete:
        return (gym_space.n,)

    else:
        return gym_space.shape


def discount(reward, gamma):
    args = [[1], [1, -gamma], reward[::-1]]
    kwargs = dict(axis=0)
    discounted_reward = scipy.signal.lfilter(*args, **kwargs)[::-1]
    return discounted_reward.astype(np.float32).ravel()


@contextmanager
def timer(key, callback=None):
    start = time.time()
    yield
    end = time.time()
    if callback:
        callback(key, end - start)

    else:
        print(key + '_time:' + str(end - start))


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val


class Buffer(object):

    def __init__(self, env, buffer_size=0):
        self.buf = self.initialize_buffer(env, buffer_size)
        self.size = buffer_size

    @staticmethod
    def initialize_buffer(env, size):
        replay_buffer = []

        for _ in range(size):
            action = env.action_space.sample()
            sars_dict = env.step(action)
            replay_buffer.append(sars_dict)

        env.reset()

        return replay_buffer

    def update(self, sars_map):
        index = rng.randint(0, self.size)
        self.buf[index] = sars_map

    def sample(self, size):
        assert self.size >= size
        return random.sample(self.buf, size)

    def flush(self):
        contents = self.buf
        self.buf = []
        return contents

    def append(self, item):
        self.buf.append(item)


class EnvWrapper(object):
    """
    problem: environments has a fair amount more information
    """

    def __init__(self, env_name, history_len=0, dtype=np.float32, **kwargs):
        self.env_ = gym.make(env_name, **kwargs)
        self.dtype = dtype
        self.state_shape = get_shape(self.env_.observation_space)
        self.action_shape = get_shape(self.env_.action_space)
        self.episode_reward = 0
        self.episode_rewards = []
        self.history = None
        self.history_len = history_len
        self.reset()

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)

        except:
            return self.env_.__getattribute__(attr)

    def reset(self):
        self.current_state = self.env_.reset().astype(self.dtype)

        zero_state = np.zeros_like(self.current_state)
        self.diff_state = zero_state

        if self.history_len:
            self.history = deque(maxlen=self.history_len)
            for _ in range(self.history_len):
                self.history.append(zero_state)

    def step(self, action):
        if isinstance(action, dict):
            action = action['action']

        init_state = self.current_state
        init_diff = self.diff_state
        new_state, reward, is_done, info = self.env_.step(action)

        new_state = self.dtype(new_state)
        reward = self.dtype(reward)
        self.episode_reward += reward

        sars_map = dict(
            initial_state=init_state,
            initial_diff=init_diff,
            action=action,
            is_done=is_done,
            reward=reward,
            new_state=new_state,
            new_diff=new_state - init_state,
            info=info,
        )
        self.current_state = new_state

        if is_done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
            self.reset()

        if self.history:
            sars_map['initial_history'] = np.concatenate(self.history, -1)
            self.history.append(new_state)
            sars_map['new_history'] = np.concatenate(self.history, -1)

        return sars_map

    def repeat(self, action, num_steps):
        accum_reward = 0
        for _ in range(num_steps):
            transition_map = self.step(action)
            accum_reward += transition_map['reward']
            if transition_map['is_done']:
                break

        transition_map['accum_reward'] = accum_reward
        return transition_map

    def policy_step(self, policy):
        action = policy(self.current_state)
        return self.step(action)

    def sample_action(self):
        return self.env_.action_space.sample()

    def sample_state(self):
        return self.dtype(self.env_.observation_space.sample())

    def sample_step(self):
        return self.step(self.sample_action())

    def start_monitor(self, trial_dir, *args, **kwargs):
        self.monitor.start(trial_dir, *args, **kwargs)
        self.reset()

    def __repr__(self):
        return ""
