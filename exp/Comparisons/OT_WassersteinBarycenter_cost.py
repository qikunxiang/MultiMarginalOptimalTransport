import numpy as np
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()

def OT_WassersteinBarycenter_cost_np(x):
    return np.squeeze(np.sum(np.square(x - np.mean(x, axis=1, keepdims=True)) / x.shape[1], axis=(1,2)))

def OT_WassersteinBarycenter_cost_tf(x):
    return tf.squeeze(tf.reduce_sum(tf.square(x - tf.reduce_mean(x, axis=1, keepdims=True)) / tf.cast(x.shape[1], tf.float64), axis=(1,2)))
