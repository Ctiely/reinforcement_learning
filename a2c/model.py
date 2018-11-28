import os
import tensorflow as tf
from layer.layers import mse, openai_entropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class A2CModel(object):
    def __init__(self, sess, policy, name="a2c_policy", num_envs=4, num_steps=5, num_stack=4, entropy_coef=0.01,
                 value_function_coeff=0.5, max_gradient_norm=0.5, optimizer_params=None):
        """
        :param policy: policy class must implement method/attribute: policy_logits, value_function
        :param optimizer_params: RMSProp params = {'learning_rate': 7e-4, 'alpha': 0.99, 'epsilon': 1e-5}
        """
        self.train_batch_size = num_envs * num_steps
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.num_stack = num_stack
        self.policy_name = name
        self.actions = None
        self.advantage = None
        self.reward = None
        self.keep_prob = None
        # self.is_training = None
        self.trainable_variables = None
        self.step_policy = None
        self.train_policy = None
        self.learning_rate_decayed = None
        self.initial_state = None
        self.X_input_step_shape = None
        self.X_input_train_shape = None
        self.policy_gradient_loss = None
        self.value_function_loss = None
        self.optimize = None
        self.entropy = None
        self.loss = None
        self.learning_rate = None
        self.num_actions = None
        self.img_height, self.img_width, self.num_channels = None, None, None

        self.policy = policy
        self.sess = sess
        self.vf_coeff = value_function_coeff
        self.entropy_coeff = entropy_coef
        self.max_grad_norm = max_gradient_norm

        if optimizer_params is None:
            self.initial_learning_rate = 7e-4
            self.alpha = 0.99
            self.epsilon = 1e-5
        elif isinstance(optimizer_params, dict):
            self.initial_learning_rate = optimizer_params['learning_rate']
            self.alpha = optimizer_params['alpha']
            self.epsilon = optimizer_params['epsilon']

    def init_input(self):
        with tf.name_scope("input"):
            self.X_input_train_shape = (None, self.img_height, self.img_width, self.num_channels * self.num_stack)
            self.X_input_step_shape = (None, self.img_height, self.img_width, self.num_channels * self.num_stack)

            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
            self.advantage = tf.placeholder(dtype=tf.float32, shape=[None], name="advantage")
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
            # self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")

    def init_network(self):
        """
        A2C model structure
        :return:
        """
        self.step_policy = self.policy(self.sess, self.X_input_step_shape, self.num_actions, self.policy_name)
        self.train_policy = self.policy(self.sess, self.X_input_train_shape, self.num_actions, self.policy_name)
        with tf.variable_scope("train_output"):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_policy.policy_logits,
                labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action, name="policy_loss")
            self.value_function_loss = tf.reduce_mean(mse(tf.squeeze(self.train_policy.value_function), self.reward),
                                                      name="value_loss")
            self.entropy = tf.reduce_mean(openai_entropy(self.train_policy.policy_logits), name="entropy")

            self.loss = self.policy_gradient_loss - self.entropy * self.entropy_coeff + \
                        self.value_function_loss * self.vf_coeff

            self.trainable_variables = tf.trainable_variables(self.policy_name)

            grads = tf.gradients(self.loss, self.trainable_variables)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            # Apply Gradients
            grads = list(zip(grads, self.trainable_variables))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                  decay=self.alpha,
                                                  epsilon=self.epsilon)
            self.optimize = optimizer.apply_gradients(grads)

    def build(self, observation_space_params, num_actions):
        self.img_height, self.img_width, self.num_channels = observation_space_params
        self.num_actions = num_actions
        self.init_input()
        self.init_network()


if __name__ == "__main__":
    from pprint import pprint
    from a2c.policies import CNNPolicy
    sess = tf.Session()
    a2c_model = A2CModel(sess, CNNPolicy, name="a2c_policy")
    a2c_model.build((84, 84, 3), 4)
    pprint(a2c_model.trainable_variables)
    writer = tf.summary.FileWriter("./graphs/test_a2c", sess.graph)
    writer.close()
    sess.close()
