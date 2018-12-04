import gym
from gym import Wrapper, wrappers
from env.atari_wrappers import wrap_deepmind


# https://github.com/MG2033/A2C/blob/master/envs/base_env.py
class BaseEnv(object):
    def __init__(self, env_name, index):
        self.env_name = env_name
        self.rank = index
        self.env = None

    def make(self):
        raise NotImplementedError("make method is not implemented")

    def step(self, actions):
        raise NotImplementedError("step method is not implemented")

    def reset(self):
        raise NotImplementedError("reset method is not implemented")

    def get_action_space(self):
        raise NotImplementedError("get_action_space method is not implemented")

    def get_observation_space(self):
        raise NotImplementedError("get_observation_space method is not implemented")

    def monitor(self, is_monitor, is_train, video_record_dir="", record_video_every=10):
        raise NotImplementedError("monitor method is not implemented")

    def render(self):
        raise NotImplementedError("render method is not implemented")


# https://github.com/MG2033/A2C/blob/master/envs/monitor.py
class Monitor(Wrapper):
    def __init__(self, env, rank=0):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.rank = rank
        self.rewards = []
        self.current_metadata = {}  # extra info that gets injected into each log entry
        self.summaries_dict = {'reward': 0, 'episode_length': 0}

    def reset(self):
        self.summaries_dict['reward'] = -1
        self.summaries_dict['episode_length'] = -1
        self.rewards = []
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.summaries_dict['reward'] = sum(self.rewards)
            self.summaries_dict['episode_length'] = len(self.rewards)
        info = self.summaries_dict
        return observation, reward, done, info

    def get_summaries_dict(self):
        return self.summaries_dict

    def monitor(self, is_monitor, is_train, record_dir="", record_video_every=10):
        if is_monitor:
            if is_train:
                self.env = wrappers.Monitor(self.env, record_dir + '_output', resume=True,
                                            video_callable=lambda count: count % record_video_every == 0)
            else:
                self.env = wrappers.Monitor(self.env, record_dir + '_test', resume=True,
                                            video_callable=lambda count: count % record_video_every == 0)
        else:
            self.env = wrappers.Monitor(self.env, record_dir + '_output', resume=True,
                                        video_callable=False)

        self.env.reset()


# https://github.com/MG2033/A2C/blob/master/envs/gym_env.py
class GymEnv(BaseEnv):
    def __init__(self, env_name, index, seed):
        super().__init__(env_name, index)
        self.seed = seed
        self.gym_env = None
        self.monitor = None
        self.make()

    def make(self):
        self.gym_env = gym.make(self.env_name)
        env = Monitor(self.gym_env, self.rank)
        self.monitor = env.monitor
        env.seed(self.seed + self.rank)
        self.env = wrap_deepmind(env)
        return env

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        self.gym_env.render()


if __name__ == "__main__":
    import random
    env = GymEnv("BreakoutNoFrameskip-v4", 0, 42)
    action_space = env.get_action_space().n
    env.reset()
    env.render()
    for _ in range(100):
        obs, r, done, info = env.step(random.randint(0, action_space - 1))
        env.render()
        if done:
            pass
