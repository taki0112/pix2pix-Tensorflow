import tensorflow as tf
import tensorflow.contrib as tf_contrib


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
"""
pad = max(k-s, 0) or max(k - n%s, 0)
size = (I-k+1+2p) // s
"""
def conv(x, channels, kernel=4, stride=2, pad=1,  scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init, strides=stride)

        return x


def deconv(x, channels, kernel=4, stride=2, scope='deconv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init, strides=stride, padding='SAME')

        return x


def resblock(x_init, channels,  scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = tf.pad(x_init, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=3, kernel_initializer=weight_init, strides=1)
            x = batch_norm(x)

        return x + x_init

def flatten(x) :
    return tf.layers.flatten(x)

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def batch_norm(x, is_training=False, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss



def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss

def discriminator_loss(real, fake):

    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = (real_loss + fake_loss) * 0.5

    return loss


def generator_loss(fake):

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss




"""
def discriminator_loss(loss_func, real, fake):
    loss = None

    if loss_func == 'wgan-gp' :
        loss = tf.reduce_mean(fake) - tf.reduce_mean(real)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))
        loss = (real_loss + fake_loss) * 0.5

    if loss_func == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        loss = (real_loss + fake_loss) * 0.5

    return loss


def generator_loss(loss_func, fake):
    loss = None

    if loss_func == 'wgan-gp' :
        loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' :
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    return loss

"""