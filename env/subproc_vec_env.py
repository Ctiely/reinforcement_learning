# https://github.com/MG2033/A2C/blob/master/envs/subproc_vec_env.py
import numpy as np
from multiprocessing import Process, Pipe


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            total_info = info.copy()  # Very important for passing by value instead of reference
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, total_info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.get_action_space(), env.get_observation_space()))
        elif cmd == 'monitor':
            is_monitor, is_train, record_dir, record_video_every = data
            env.monitor(is_monitor, is_train, record_dir, record_video_every)
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):
    #通过remote.send将信息发送至remote.recv()里,进行操作后将新信息再次发送到remote.recv()里
    def __init__(self, env_fns):
        """
        env_fns: list of environments function wrapper to run in sub-processes
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for work_remote, env_fn in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None)) #发送信息到remote.recv()里

        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        """
        actions的维度是环境个数,
        :return: 返回所有环境的信息
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
        重启所有环境
        :return: 返回所有环境的初始状态
        """
        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        """
        关闭通信,阻塞进程
        :return: None
        """
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def monitor(self, is_monitor=True, is_train=True, record_dir="", record_video_every=10):
        for remote in self.remotes:
            remote.send(('monitor', (is_monitor, is_train, record_dir, record_video_every)))

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    @property
    def num_envs(self):
        return len(self.remotes)


if __name__ == "__main__":
    from env.envs import GymEnv
    # The reason behind this design pattern is to pass the function handler when required after serialization.
    def make_env(index, seed):
        def __make_env(): # 需要包一层函数,否则会报'GymEnv' object is not callable错误
            env = GymEnv("BreakoutNoFrameskip-v4", index, seed)
            return env
        return __make_env
    envs = SubprocVecEnv([make_env(i, 42) for i in range(4)]) # envs里包含了四个游戏环境
    envs.reset()
    print(envs.num_envs)
    obs, rewards, dones, infos = envs.step([0, 1, 3, 2])
    print(obs.shape)
    print(rewards, dones)
    print(len(infos))
    envs.close()