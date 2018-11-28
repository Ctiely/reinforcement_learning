import os
import time
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from env.Logger import EnvSummaryLogger
from utils.utils import create_list_dirs
from utils.lr_decay import LearningRateDecay
logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BaseTrainer(object):
    def __init__(self, sess, model, max_to_keep=5, summary_dir="./graphs"):
        self.sess = sess
        self.model = model

        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.summary_placeholders = {}
        self.summary_ops = {}

    def save(self, checkpoint_dir):
        logging.info("Saving model...")
        self.saver.save(self.sess, checkpoint_dir, self.global_step_tensor)
        logging.info("Model saved.")

    def _load_model(self, checkpoint_dir):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logging.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info("model loaded\n\n")
        else:
            logging.warning("No model available!\n\n")

    def _init_trainer(self):
        # init the global step, global time step, the current epoch and the summaries
        self.__init_global_step()
        self.__init_global_time_step()
        self.__init_cur_epoch()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        logging.info("init trainer...")
        self.sess.run(self.init)

    def __init_cur_epoch(self):
        """
        Create cur epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

    def __init_global_time_step(self):
        """
        Create a global time step variable to be a reference to the number of time steps
        :return:
        """
        with tf.variable_scope('global_time_step'):
            self.global_time_step_tensor = tf.Variable(0, trainable=False, name='global_time_step')
            self.global_time_step_input = tf.placeholder('int32', None, name='global_time_step_input')
            self.global_time_step_assign_op = self.global_time_step_tensor.assign(self.global_time_step_input)


class A2CTrainer(BaseTrainer):
    def __init__(self, sess, a2c_model, num_iterations=10e8, learning_rate=7e-4, reward_discount_rate=0.99, lr_decay_method='linear',
                 max_to_keep=5, save_step=20000, summary_dir="./graphs/A2C", checkpoint_dir=None):
        super().__init__(sess, a2c_model, max_to_keep, summary_dir)
        self.num_steps = self.model.num_steps
        self.save_step = save_step
        self.checkpoint_dir = checkpoint_dir

        self.cur_step = 0
        self.cur_iteration = 0
        self.global_time_step = 0
        self.observation_s = None
        self.states = None
        self.dones = None

        self.envs = None
        self.num_envs = self.model.num_envs
        self.num_iterations = int(num_iterations)

        self.gamma = reward_discount_rate

        self.learning_rate_decayed = LearningRateDecay(v=learning_rate,
                                                       nvalues=self.num_iterations * self.model.num_steps * self.num_envs,
                                                       lr_decay_method=lr_decay_method)

        self.env_summary_logger = EnvSummaryLogger(sess, create_list_dirs(summary_dir, 'env', self.num_envs))

    def train(self, envs):
        """
        :param envs: (SubprocVecEnv)
        """
        assert self.model.num_envs == envs.num_envs
        self._init_trainer()
        if self.checkpoint_dir is not None:
            self._load_model(self.checkpoint_dir)
        else:
            self.checkpoint_dir = "./checkpoints/"

        self.envs = envs
        self.observation_s = np.zeros(
            (envs.num_envs, self.model.img_height, self.model.img_width, self.model.num_channels * self.model.num_stack),
            dtype=np.uint8)
        self.observation_s = self.__observation_stack(envs.reset(), self.observation_s)
        self.dones = [False for _ in range(self.envs.num_envs)]

        tstart = time.time()
        loss_list = np.zeros(100, )
        policy_loss_list = np.zeros(100, )
        value_loss_list = np.zeros(100, )
        policy_entropy_list = np.zeros(100, )
        fps_list = np.zeros(100, )
        arr_idx = 0
        start_step = self.global_step_tensor.eval(self.sess)

        self.global_time_step = self.global_time_step_tensor.eval(self.sess)
        for step in tqdm(range(start_step, self.num_iterations + 1, 1), initial=start_step,
                         total=self.num_iterations):
            self.cur_step = step
            # 收集一次数据
            obs, rewards, actions, values = self.__rollout()
            # 更新一次参数
            loss, policy_loss, value_loss, policy_entropy = self.__update(obs, rewards, actions, values)

            loss_list[arr_idx] = loss
            nseconds = time.time() - tstart
            fps_list[arr_idx] = int((step * self.num_steps * self.envs.num_envs) / nseconds)
            policy_loss_list[arr_idx] = policy_loss
            value_loss_list[arr_idx] = value_loss
            policy_entropy_list[arr_idx] = policy_entropy

            if step % self.save_step == 0:
                self.save(self.checkpoint_dir)

            # Update the Global step
            self.global_step_assign_op.eval(session=self.sess, feed_dict={
                self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
            arr_idx += 1

            if not arr_idx % 100:
                mean_loss = np.mean(loss_list)
                mean_policy_loss = np.mean(policy_loss_list)
                mean_value_loss = np.mean(value_loss_list)
                mean_fps = np.mean(fps_list)
                mean_pe = np.mean(policy_entropy_list)
                logging.info(
                    "\nstep: {}, total loss: {:.5f}, policy loss: {:.5f}, value loss: {:.5f}, entropy: {:.5f},fps: {}"
                    .format(step, mean_loss, mean_policy_loss, mean_value_loss, mean_pe, mean_fps))
                arr_idx = 0
        self.envs.close()

    def test(self, env, total_timesteps=10000):
        self._init_trainer()
        if self.checkpoint_dir is not None:
            self._load_model(self.checkpoint_dir)
        else:
            logging.error("checkpoint_dir is None, you must run train before run test OR specify a checkpoint_dir.")

        observation_s = np.zeros(
            (env.num_envs, self.model.img_height, self.model.img_width, self.model.num_channels * self.model.num_stack),
            dtype=np.uint8)
        observation_s = self.__observation_stack(env.reset(), observation_s)

        for _ in tqdm(range(int(total_timesteps))):
            actions, values = self.model.step_policy.step(observation_s)
            observation, rewards, dones, _ = env.step(actions)
            for n, done in enumerate(dones):
                if done:
                    observation_s[n] *= 0
            observation_s = self.__observation_stack(observation, observation_s)
        env.close()

    def __observation_stack(self, new_observation, old_observation):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        updated_observation = np.roll(old_observation, shift=-1, axis=3)
        updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
        return updated_observation

    def __discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        # Start from downwards to upwards like Bellman backup operation.
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)

        return discounted[::-1]

    def __rollout(self):
        train_input_shape = (self.model.train_batch_size, self.model.img_height, self.model.img_width,
                             self.model.num_channels * self.model.num_stack)
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        for n in range(self.num_steps):
            # step_policy进行游戏
            actions, values = self.model.step_policy.step(self.observation_s) # 返回最优行动以及行动的价值[envs.num_env,]

            mb_obs.append(np.copy(self.observation_s))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take a step in the real environment
            observations, rewards, dones, infos = self.envs.step(actions)

            # Tensorboard dump, divided by 100 to rescale (to make the steps make sense)
            self.env_summary_logger.add_summary_all(int(self.global_time_step / 100), infos)
            self.global_time_step += 1
            self.global_time_step_assign_op.eval(session=self.sess,
                                                 feed_dict={self.global_time_step_input: self.global_time_step})
            self.dones = dones
            for n, done in enumerate(dones):
                # n代表着环境索引
                if done:
                    self.observation_s[n] *= 0
            self.observation_s = self.__observation_stack(observations, self.observation_s)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Conversion from (time_steps, num_envs) to (num_envs, time_steps)
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(train_input_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:] # 保留的是下一个状态是否done
        last_values = self.model.step_policy.value(self.observation_s).tolist()

        # Discount/bootstrap off value fn in all parallel environments
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.__discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # Instead of (num_envs, time_steps). Make them num_envs * time_steps.
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()

        return mb_obs, mb_rewards, mb_actions, mb_values

    def __update(self, mb_obs, mb_rewards, mb_actions, mb_values):
        # Updates the model per trajectory for using parallel environments. Uses the train_policy.
        # 使用train_policy更新参数
        advantages = mb_rewards - mb_values
        current_learning_rate = None
        for step in range(len(mb_obs)):
            current_learning_rate = self.learning_rate_decayed.value()
        feed_dict = {self.model.train_policy.X_input: mb_obs,
                     self.model.actions: mb_actions,
                     self.model.advantage: advantages,
                     self.model.reward: mb_rewards,
                     self.model.learning_rate: current_learning_rate}
        loss, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.model.loss, self.model.policy_gradient_loss, self.model.value_function_loss, self.model.entropy,
             self.model.optimize],
            feed_dict=feed_dict)

        return loss, policy_loss, value_loss, policy_entropy


if __name__ == "__main__":
    from utils.utils import make_env
    from a2c.model import A2CModel
    from a2c.policies import CNNPolicy
    from env.subproc_vec_env import SubprocVecEnv

    envs = SubprocVecEnv([make_env(i) for i in range(3)])
    sess = tf.Session()
    a2c_model = A2CModel(sess, CNNPolicy, name="a2c_model", num_envs=envs.num_envs)
    logging.info("Build Model...")
    a2c_model.build(envs.observation_space.shape, envs.action_space.n)
    a2c_trainer = A2CTrainer(sess, a2c_model, num_iterations=100, save_step=100)
    a2c_trainer.train(envs)
    envs = SubprocVecEnv([make_env(0, 42)])
    a2c_trainer.test(envs)
    sess.close()