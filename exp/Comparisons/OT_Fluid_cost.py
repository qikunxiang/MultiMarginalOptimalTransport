import numpy as np
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()

def OT_Fluid_cost_np(x, arrangement):
    return np.squeeze(np.sum(np.square(x[:, 0:-1, :] - x[:, 1:, :]), axis=1) + np.square(x[:, -1, :] - arrangement(x[:, 0, :])))

def OT_Fluid_arrangement0_np(x):
    return x

def OT_Fluid_arrangement1_np(x):
    return np.minimum(2 * x, 2 - 2 * x)

def OT_Fluid_arrangement2_np(x):
    return 4 * np.minimum(np.abs(x - 1/4), np.abs(x - 3/4))

def OT_Fluid_cost_tf(x, arrangement):
    return tf.squeeze(tf.reduce_sum(tf.math.square(x[:, 0:-1] - x[:, 1:]), axis=1) + tf.math.square(x[:, -1] - arrangement(x[:, 0])))

def OT_Fluid_arrangement0_tf(x):
    return x

def OT_Fluid_arrangement1_tf(x):
    return tf.math.minimum(2 * x, 2 - 2 * x)

def OT_Fluid_arrangement2_tf(x):
    return 4 * tf.math.minimum(tf.abs(x - 1/4), tf.abs(x - 3/4))