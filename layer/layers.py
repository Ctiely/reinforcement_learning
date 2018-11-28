import os
# https://github.com/MG2033/A2C/blob/master/layers.py
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def noise_and_argmax(logits):
    # Add noise then take the argmax
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def openai_entropy(logits):
    # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def softmax_entropy(p0):
    # Normal information theory entropy by Shannon
    return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis=1)


def mse(predicted, ground_truth):
    # Mean-squared error
    return tf.square(predicted - ground_truth) / 2.


def orthogonal_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # Orthogonal Initializer that uses SVD. The unused variables are just for passing in tensorflow
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable(name="weights", shape=kernel_shape, dtype=tf.float32, initializer=initializer)

    collection = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name="w_loss")
        tf.add_to_collection(collection, weight_decay)
    variable_summaries(w)
    return w


def variable_summaries(variable):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param variable: variable to be summarized
    :return: None
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar("mean", mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(variable))
        tf.summary.scalar('min', tf.reduce_min(variable))
        tf.summary.histogram('histogram', variable)


def max_pool_2d(x, size=(2, 2)):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1],
                          strides=[1, size_x, size_y, 1], padding='VALID', name='pooling')


def flatten(x, name="flatten"):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first], name=name)
    return o


# convolution layer
def conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
             initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope("layer_weights"):
            if w is None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope("layer_biases"):
            if isinstance(bias, float):
                bias = tf.get_variable(name="biases", shape=[num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope("layer_conv2d"):
            conv = tf.nn.conv2d(input=x, filter=w, strides=stride, padding=padding)
            out = tf.nn.bias_add(conv, bias)

        return out


def atrous_conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', dilation_rate=1,
                    initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Atrous Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param dilation_rate: (integer) The amount of dilation required. If equals 1, it means normal convolution.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_atrous_conv2d'):
            conv = tf.nn.atrous_conv2d(x, w, dilation_rate, padding)
            out = tf.nn.bias_add(conv, bias)

        return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                            padding=padding, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
        else:
            conv_o_bn = conv_o_b

        if activation is None:
            conv_a = conv_o_bn
        else:
            conv_a = activation(conv_o_bn)

        if dropout_keep_prob != -1:
            conv_dr = tf.nn.dropout(conv_a, keep_prob=dropout_keep_prob)
        else:
            conv_dr = conv_a

        if max_pool_enabled:
            conv_o = max_pool_2d(conv_dr)
        else:
            conv_o = conv_dr

        return conv_o


# dense layer
def dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
            bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        variable_summaries(bias)
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0, activation=None, batchnorm_enabled=False, dropout_keep_prob=-1, is_training=True):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = dense_p(scope, x=x, w=w, output_dim=output_dim, initializer=initializer,
                            l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training)
        else:
            dense_o_bn = dense_o_b

        if activation is None:
            dense_a = dense_o_bn
        else:
            dense_a = activation(dense_o_bn)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.nn.dropout(dense_a, keep_prob=dropout_keep_prob)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr

        return dense_o


if __name__ == "__main__":
    from pprint import pprint
    x = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 3], name="input")
    with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
        conv1 = conv2d("conv1", x)
        conv2 = conv2d("conv2", conv1)
        conv3 = conv2d("conv3", conv2)
        flatten_conv3 = flatten(conv3)
        fc = dense("dense", flatten_conv3)

    with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
        conv1_ = conv2d("conv1", x)
        conv2_ = conv2d("conv2", conv1_)
        conv3_ = conv2d("conv3", conv2_)
        flatten_conv3_ = flatten(conv3_)
        fc_ = dense("dense", flatten_conv3_)

    sess = tf.Session()
    writer = tf.summary.FileWriter("./graphs/test_layers", sess.graph)
    pprint(tf.trainable_variables()) # 两个网络的名字虽然不一样,但是它们内部用的是同一套参数
    sess.close()
