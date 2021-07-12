import tensorflow as tf
import math

class SDR(tf.keras.losses.Loss):

    def __init__(self):
        super(SDR, self).__init__(name='SDR')
        self.epsilon = 1e-7

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s - s_hat) / (tf.norm(s) + self.epsilon) + self.epsilon) / math.log(10)
