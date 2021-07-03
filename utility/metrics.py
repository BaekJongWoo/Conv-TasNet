import tensorflow as tf
import math

def SNR(s, s_hat):
    epsilon = 1e-7
    s_target = tf.math.reduce_sum(tf.math.multiply(s_hat, s)) * s / (tf.math.pow(tf.norm(s), 2) + epsilon)
    e_noise = s_hat - s_target
    return 20 * tf.math.log(tf.norm(s_target) / (tf.norm(e_noise) + epsilon) + epsilon) / math.log(10)