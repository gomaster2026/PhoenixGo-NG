#!/usr/bin/env python3
#
#    This file is part of Leela Zero.
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Zero is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Zero is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def float32_variable_storage_getter(getter, name, shape=None, dtype= None,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored
    in float32, even when the training computations are done in float16.

    This is needed for mixed-precision (fp16) training to avoid gradient
    underflow: the master copy of the weights is kept in fp32, while the
    actual computation uses fp16 copies.
    """
    if dtype == tf.float16:
        var = getter(name, shape, tf.float32, *args, **kwargs)
        return tf.cast(var, dtype)
    else:
        return getter(name, shape, dtype, *args, **kwargs)


class LossScalingOptimizer:
    """Optimizer wrapper that scales the loss to prevent gradient underflow
    when using float16.

    Multiplies the loss by `scale` before computing gradients, then divides
    the gradients by `scale` before applying them. This keeps gradients in
    a representable range for fp16.
    """
    def __init__(self, opt, scale=1):
        self._opt = opt
        self._scale = scale

    def compute_gradients(self, loss, *args, **kwargs):
        """Compute gradients of scaled loss."""
        scaled_loss = loss * self._scale
        gvs = self._opt.compute_gradients(scaled_loss, *args, **kwargs)
        scaled_gvs = []
        for g, v in gvs:
            if g is not None:
                # Unscale the gradient back
                scaled_gvs.append((g / self._scale, v))
            else:
                scaled_gvs.append((g, v))
        return scaled_gvs

    def apply_gradients(self, *args, **kwargs):
        """Apply gradients using the wrapped optimizer."""
        return self._opt.apply_gradients(*args, **kwargs)
