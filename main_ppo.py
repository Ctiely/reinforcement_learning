if __name__ == "__main__":
    import tensorflow as tf
    from ppo.PPO import PPO
    from utils.utils import make_env
    from env.subproc_vec_env import SubprocVecEnv
    
    envs = SubprocVecEnv([make_env(i) for i in range(4)])
    sess = tf.Session()
    ppo = PPO(sess, envs, num_iterations=1000, save_step=1000, checkpoint_dir="./ppo/checkpoints/")
    ppo.train()
    env = SubprocVecEnv([make_env(0)])
    ppo.test(env, 100000)
    sess.close()