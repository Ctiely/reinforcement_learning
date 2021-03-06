if __name__ == "__main__":
    import tensorflow as tf
    from a2c.A2C import A2C
    from utils.utils import make_env
    from env.subproc_vec_env import SubprocVecEnv

    envs = SubprocVecEnv([make_env(i) for i in range(4)])
    sess = tf.Session()
    a2c = A2C(sess, envs, num_iterations=1000, save_step=1000, checkpoint_dir="./a2c/checkpoints/")
    a2c.train()
    env = SubprocVecEnv([make_env(0)])
    a2c.test(env, 100000)
    sess.close()
