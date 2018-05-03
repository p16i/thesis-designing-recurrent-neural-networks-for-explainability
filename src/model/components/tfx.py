import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

import numbers


def dropout_with_mark_returned(x, keep_prob, noise_shape=None, seed=None, name=None, binary_mark_tensor=None):
    with tf.name_scope(name, "dropout", [x]) as name:
        x = tf.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = tf.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        if binary_mark_tensor is None:
            noise_shape = noise_shape if noise_shape is not None else tf.shape(x)
            random_tensor = keep_prob
            random_tensor += tf.random_uniform(
                noise_shape, seed=seed, dtype=x.dtype)
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            binary_mark_tensor = tf.floor(random_tensor)
        ret = tf.div(x, keep_prob) * binary_mark_tensor

        return ret, binary_mark_tensor
