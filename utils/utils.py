import os
import random
import numpy as np
import tensorflow as tf
from env.envs import GymEnv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_env(index, env_name="BreakoutNoFrameskip-v4", seed=42):
    def __make_env():
        env = GymEnv(env_name, index, seed)
        return env

    return __make_env


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = "experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    output_dir = experiment_dir + 'output/'
    test_dir = experiment_dir + 'test/'
    dirs = [summary_dir, checkpoint_dir, output_dir, test_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created")
        return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def create_list_dirs(input_dir, prefix_name, count):
    dirs_path = []
    for i in range(count):
        dirs_path.append(input_dir + prefix_name + '-' + str(i))
        create_dirs([input_dir + prefix_name + '-' + str(i)])
    return dirs_path


def set_all_global_seeds(i):
    try:
        tf.set_random_seed(i)
        np.random.seed(i)
        random.seed(i)
    except:
        return ImportError


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
