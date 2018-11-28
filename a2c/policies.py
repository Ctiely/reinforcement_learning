import os
import numpy as np
import tensorflow as tf
from layer.layers import conv2d, dense, flatten, orthogonal_initializer, noise_and_argmax
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BasePolicy(object):
    def __init__(self, sess, input_shape):
        self.sess = sess
        self.input_shape = input_shape
        self.X_input = None
        self.value_s = None
        self.action_s = None
        self.policy_logits = None
        self.value_function = None

    def step(self, observation):
        raise NotImplementedError("step method not implemented")

    def value(self, observation):
        raise NotImplementedError("value method not implemented")


class CNNPolicy(BasePolicy):
    def __init__(self, sess, input_shape, num_actions, policy_name):
        super().__init__(sess, input_shape)

        with tf.name_scope("policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)

        with tf.variable_scope(policy_name, reuse=tf.AUTO_REUSE):
            conv1 = conv2d('conv1', tf.cast(self.X_input, tf.float32) / 255., num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4), initializer=orthogonal_initializer(np.sqrt(2)),
                           activation=tf.nn.relu)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu)

            conv3_flattened = flatten(conv3)

            fc4 = dense('fc4', conv3_flattened, output_dim=512,
                        initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu)

            self.policy_logits = dense("policy_logits", fc4, output_dim=num_actions,
                                       initializer=orthogonal_initializer(1.0))
            self.value_function = dense("value_function", fc4, output_dim=1,
                                        initializer=orthogonal_initializer(1.0))

            with tf.name_scope("value"):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope("policy"):
                self.action_s = noise_and_argmax(self.policy_logits)

    def step(self, observation):
        # Take a step using the model and return the predicted policy and value function
        action, value = self.sess.run([self.action_s, self.value_s], feed_dict={self.X_input: observation})
        return action, value

    def value(self, observation):
        # Return the predicted value function for a given observation
        return self.sess.run(self.value_s, feed_dict={self.X_input: observation})


if __name__ == "__main__":
    obs = np.zeros((4, 84, 84, 3))
    sess = tf.Session()
    policy = CNNPolicy(sess, (None, 84, 84, 3), 4, "policy")
    print(policy.value_function, policy.value_s)
    sess.run(tf.global_variables_initializer())
    print(policy.step(obs))
    print(policy.value(obs))
    writer = tf.summary.FileWriter("./graphs/test_policy", sess.graph)
    writer.close()
    sess.close()