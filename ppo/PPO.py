import logging
from ppo.model import PPOModel
from policies import CNNPolicy
from trainer import PPOTrainer
from utils.utils import set_all_global_seeds
logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')


class PPO(object):
    def __init__(self, sess, envs,
                 seed=42,
                 num_iterations=1e6,
                 learning_rate=7e-4,
                 nminibatches=4,
                 lam=0.95,
                 noptepochs=4,
                 reward_discount_rate=0.99,
                 lr_decay_method='linear',
                 max_to_keep=5,
                 save_step=20000,
                 summary_dir="./graphs/PPO",
                 checkpoint_dir=None, # trainer
                 name="ppo_policy",
                 num_steps=5,
                 num_stack=4,
                 entropy_coef=0.01,
                 value_function_coeff=0.5,
                 clip_ratio=0.2,
                 max_gradient_norm=0.5,
                 optimizer_params=None): # policy
        self.sess = sess
        self.envs = envs
        self.ppo_model = PPOModel(sess=self.sess,
                                  policy=CNNPolicy,
                                  name=name,
                                  clip_ratio=clip_ratio,
                                  num_envs=envs.num_envs,
                                  num_steps=num_steps,
                                  num_stack=num_stack,
                                  entropy_coef=entropy_coef,
                                  value_function_coeff=value_function_coeff,
                                  max_gradient_norm=max_gradient_norm,
                                  optimizer_params=optimizer_params)
        logging.info("BUILD PPO MODEL...")
        self.ppo_model.build(self.envs.observation_space.shape, self.envs.action_space.n)
        self.ppo_trainer = PPOTrainer(sess=self.sess,
                                      ppo_model=self.ppo_model,
                                      num_iterations=num_iterations,
                                      nminibatches=nminibatches,
                                      lam=lam,
                                      noptepochs=noptepochs,
                                      learning_rate=learning_rate,
                                      reward_discount_rate=reward_discount_rate,
                                      lr_decay_method=lr_decay_method,
                                      max_to_keep=max_to_keep,
                                      save_step=save_step,
                                      summary_dir=summary_dir,
                                      checkpoint_dir=checkpoint_dir)
        set_all_global_seeds(seed)

    def train(self):
        logging.info("TRAINING...")
        self.ppo_trainer.train(self.envs)
        logging.info("DONE...")

    def test(self, env, total_timesteps=10000, record_dir="./records"):
        assert env.num_envs == 1
        env.monitor(is_monitor=True, is_train=False, record_dir=record_dir, record_video_every=20)
        logging.info("TESTING...")
        self.ppo_trainer.test(env=env, total_timesteps=total_timesteps)
        logging.info("DONE...")
