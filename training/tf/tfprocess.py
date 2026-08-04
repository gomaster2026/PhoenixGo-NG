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

import math
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import unittest

from mixprec import float32_variable_storage_getter, LossScalingOptimizer


def weight_variable(name, shape, dtype):
    """Xavier initialization"""
    stddev = np.sqrt(2.0 / (sum(shape)))
    # Do not use a constant as the initializer, that will cause the
    # variable to be stored in wrong dtype.
    weights = tf.get_variable(
        name, shape, dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
# We do not regularlize biases, so they are not
# added to the regularlizer collection
def bias_variable(name, shape, dtype):
    bias = tf.get_variable(name, shape, dtype=dtype,
                           initializer=tf.zeros_initializer())
    return bias


def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW',
                        strides=[1, 1, 1, 1], padding='SAME')

# Restore session from checkpoint. It silently ignore mis-matches
# between the checkpoint and the graph. Specifically
# 1. values in the checkpoint for which there is no corresponding variable.
# 2. variables in the graph for which there is no specified value in the
#    checkpoint.
# 3. values where the checkpoint shape differs from the variable shape.
# (variables without a value in the checkpoint are left at their default
# initialized value)
def optimistic_restore(session, save_file, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
         if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars)
    opt_saver.restore(session, save_file)

# Class holding statistics
class Stats:
    def __init__(self):
        self.s = {}
    def add(self, stat_dict):
        for (k,v) in stat_dict.items():
            if k not in self.s:
                self.s[k] = []
            self.s[k].append(v)
    def n(self, name):
        return len(self.s[name] or [])
    def mean(self, name):
        return np.mean(self.s[name] or [0])
    def stddev_mean(self, name):
        # standard deviation in the sample mean.
        return math.sqrt(
            np.var(self.s[name] or [0]) / max(0.0001, (len(self.s[name]) - 1)))
    def str(self):
        return ', '.join(
            ["{}={:g}".format(k, np.mean(v or [0])) for k,v in self.s.items()])
    def clear(self):
        self.s = {}
    def summaries(self, tags):
        return [tf.Summary.Value(
            tag=k, simple_value=self.mean(v)) for k,v in tags.items()]

# Simple timer
class Timer:
    def __init__(self):
        self.last = time.time()
    def elapsed(self):
        # Return time since last call to 'elapsed()'
        t = time.time()
        e = t - self.last
        self.last = t
        return e

class TFProcess:
    def __init__(self, residual_blocks, residual_filters):
        # Network structure
        self.residual_blocks = residual_blocks
        self.residual_filters = residual_filters

        # model type: full precision (fp32) or mixed precision (fp16)
        self.model_dtype = tf.float32

        # Scale the loss to prevent gradient underflow
        self.loss_scale = 1 if self.model_dtype == tf.float32 else 128

        # L2 regularization parameter applied to weights.
        self.l2_scale = 1e-4

        # Set number of GPUs for training
        self.gpus_num = 1

        # For exporting
        self.weights = []

        # Output weight file with averaged weights
        self.swa_enabled = True

        # Net sampling rate (e.g 2 == every 2nd network).
        self.swa_c = 1

        # Take an exponentially weighted moving average over this
        # many networks. Under the SWA assumptions, this will reduce
        # the distance to the optimal value by a factor of 1/sqrt(n)
        self.swa_max_n = 16

        # Recalculate SWA weight batchnorm means and variances
        self.swa_recalc_bn = True

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.session = tf.Session(config=config)

        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def init(self, batch_size, macrobatch=1, gpus_num=None, logbase='leelalogs'):
        self.batch_size = batch_size
        self.macrobatch = macrobatch
        self.logbase = logbase
        # Input batch placeholders
        self.planes = tf.placeholder(tf.string, name='in_planes')
        self.probs = tf.placeholder(tf.string, name='in_probs')
        self.winner = tf.placeholder(tf.string, name='in_winner')

        # Mini-batches come as raw packed strings. Decode
        # into tensors to feed into network.
        planes = tf.decode_raw(self.planes, tf.uint8)
        probs = tf.decode_raw(self.probs, tf.float32)
        winner = tf.decode_raw(self.winner, tf.float32)

        planes = tf.cast(planes, self.model_dtype)

        planes = tf.reshape(planes, (batch_size, 17, 19*19))
        probs = tf.reshape(probs, (batch_size, 19*19 + 1))
        winner = tf.reshape(winner, (batch_size, 1))

        if gpus_num is None:
            gpus_num = self.gpus_num
        self.init_net(planes, probs, winner, gpus_num)

    def init_net(self, planes, probs, winner, gpus_num):
        self.y_ = probs   # (tf.float32, [None, 362])
        self.sx = tf.split(planes, gpus_num)
        self.sy_ = tf.split(probs, gpus_num)
        self.sz_ = tf.split(winner, gpus_num)
        self.batch_norm_count = 0
        self.reuse_var = None

        # You need to change the learning rate here if you are training
        # from a self-play training set, for example start with 0.005 instead.
        opt = tf.train.MomentumOptimizer(
            learning_rate=0.05, momentum=0.9, use_nesterov=True)

        opt = LossScalingOptimizer(opt, scale=self.loss_scale)

        # Construct net here.
        tower_grads = []
        tower_loss = []
        tower_policy_loss = []
        tower_mse_loss = []
        tower_reg_term = []
        tower_y_conv = []
        with tf.variable_scope("fp32_storage",
                               # this forces trainable variables to be stored as fp32
                               custom_getter=float32_variable_storage_getter):
            for i in range(gpus_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        loss, policy_loss, mse_loss, reg_term, y_conv = self.tower_loss(
                            self.sx[i], self.sy_[i], self.sz_[i])

                        # Reset batchnorm key to 0.
                        self.reset_batchnorm_key()

                        tf.get_variable_scope().reuse_variables()
                        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                            grads = opt.compute_gradients(loss)

                        tower_grads.append(grads)
                        tower_loss.append(loss)
                        tower_policy_loss.append(policy_loss)
                        tower_mse_loss.append(mse_loss)
                        tower_reg_term.append(reg_term)
                        tower_y_conv.append(y_conv)

        # Average gradients from different GPUs
        self.loss = tf.reduce_mean(tower_loss)
        self.policy_loss = tf.reduce_mean(tower_policy_loss)
        self.mse_loss = tf.reduce_mean(tower_mse_loss)
        self.reg_term = tf.reduce_mean(tower_reg_term)
        self.y_conv = tf.concat(tower_y_conv, axis=0)
        self.mean_grads = self.average_gradients(tower_grads)

        # Do swa after we contruct the net
        if self.swa_enabled is True:
            # Count of networks accumulated into SWA
            self.swa_count = tf.Variable(0., name='swa_count', trainable=False)
            # Count of networks to skip
            self.swa_skip = tf.Variable(self.swa_c, name='swa_skip',
                trainable=False)
            # Build the SWA variables and accumulators
            accum=[]
            load=[]
            n = self.swa_count
            for w in self.weights:
                name = w.name.split(':')[0]
                var = tf.Variable(
                    tf.zeros(shape=w.shape), name='swa/'+name, trainable=False)
                accum.append(
                    tf.assign(var, var * (n / (n + 1.)) + w * (1. / (n + 1.))))
                load.append(tf.assign(w, var))
            with tf.control_dependencies(accum):
                self.swa_accum_op = tf.assign_add(n, 1.)
            self.swa_load_op = tf.group(*load)

        # Accumulate gradients
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        total_grad=[]
        grad_ops=[]
        clear_var=[]
        self.grad_op_real = self.mean_grads
        for (g, v) in self.grad_op_real:
            if g is None:
                total_grad.append((g,v))
                continue
            name = v.name.split(':')[0]
            gsum = tf.get_variable(name='gsum/'+name,
                                   shape=g.shape,
                                   trainable=False,
                                   initializer=tf.zeros_initializer)
            total_grad.append((gsum, v))
            grad_ops.append(tf.assign_add(gsum, g))
            clear_var.append(gsum)
        # Op to compute gradients and add to running total in 'gsum/'
        self.grad_op = tf.group(*grad_ops)

        # Op to apply accmulated gradients
        self.train_op = opt.apply_gradients(total_grad)

        zero_ops = []
        for g in clear_var:
            zero_ops.append(
                tf.assign(g, tf.zeros(shape=g.shape, dtype=g.dtype)))
        # Op to clear accumulated gradients
        self.clear_op = tf.group(*zero_ops)

        # Op to increment global step counter
        self.step_op = tf.assign_add(self.global_step, 1)

        correct_prediction = \
            tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Summary part
        self.test_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(),
                         self.logbase + "/test"), self.session.graph)
        self.train_writer = tf.summary.FileWriter(
            os.path.join(os.getcwd(),
                         self.logbase + "/train"), self.session.graph)

        # Build checkpoint saver
        self.saver = tf.train.Saver()

        # Initialize all variables
        self.session.run(tf.global_variables_initializer())

    def average_gradients(self, tower_grads):
        # Average gradients from different GPUs
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, dim=0)
                grads.append(expanded_g)

            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, reduction_indices=0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def tower_loss(self, x, y_, z_):
        y_conv, z_conv = self.construct_net(x)

        # Cast the nn result back to fp32 to avoid loss overflow/underflow
        if self.model_dtype != tf.float32:
            y_conv = tf.cast(y_conv, tf.float32)
            z_conv = tf.cast(z_conv, tf.float32)

        # Calculate loss on policy head
        cross_entropy = \
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                    logits=y_conv)
        policy_loss = tf.reduce_mean(cross_entropy)

        # Loss on value head
        mse_loss = \
            tf.reduce_mean(tf.squared_difference(z_, z_conv))

        # Regularizer
        reg_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
        reg_term = self.l2_scale * tf.add_n(
            [tf.cast(tf.nn.l2_loss(v), tf.float32) for v in reg_variables])

        # For training from a (smaller) dataset of strong players, you will
        # want to reduce the factor in front of self.mse_loss here.
        loss = 1.0 * policy_loss + 1.0 * mse_loss + reg_term

        return loss, policy_loss, mse_loss, reg_term, y_conv

    def assign(self, var, values):
        try:
            self.session.run(tf.assign(var, values))
        except:
            print("Failed to assign {}: var shape {}, values shape {}".format(
                var.name, var.shape, values.shape))
            raise

    def replace_weights(self, new_weights):
        # v3 PhoenixGo format:
        #   self.weights order per conv+BN unit: conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var
        #   file order matches (6 lines per unit)
        #   trunk BN: gamma, beta, mean, var (4 lines, no conv)
        #   fc layers: w, b (2 lines)
        # BN moving_variance in file is raw var; TF stores moving_variance as-is.
        # No folding needed — TF batch_normalization uses gamma/beta/mean/var directly.
        for e, weights in enumerate(self.weights):
            if isinstance(weights, str):
                weights = tf.get_default_graph().get_tensor_by_name(weights)
            if weights.name.endswith('/batch_normalization/gamma:0'):
                # gamma: direct copy
                new_w = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_w)
            elif weights.name.endswith('/batch_normalization/beta:0'):
                # beta: direct copy (v3 has explicit beta, no folding)
                new_w = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_w)
            elif weights.name.endswith('/batch_normalization/moving_mean:0'):
                new_w = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_w)
            elif weights.name.endswith('/batch_normalization/moving_variance:0'):
                # moving_variance: direct copy (TF applies epsilon internally)
                new_w = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_w)
            elif weights.shape.ndims == 4:
                # Convolution weights need a transpose
                # TF (kYXInputOutput) [fh, fw, in, out]
                # Leela/cuDNN (kOutputInputYX) [out, in, fh, fw]
                s = weights.shape.as_list()
                shape = [s[i] for i in [3, 2, 0, 1]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [2, 3, 1, 0]))
            elif weights.shape.ndims == 2:
                # Fully connected: TF [in, out] ← Leela [out, in]
                s = weights.shape.as_list()
                shape = [s[i] for i in [1, 0]]
                new_weight = tf.constant(new_weights[e], shape=shape)
                self.assign(weights, tf.transpose(new_weight, [1, 0]))
            else:
                # Biases (conv_b, fc_b) — direct copy
                new_weight = tf.constant(new_weights[e], shape=weights.shape)
                self.assign(weights, new_weight)

    def restore(self, file):
        print("Restoring from {0}".format(file))
        optimistic_restore(self.session, file)

    def measure_loss(self, batch, training=False):
        # Measure loss over one batch. If training is true, also
        # accumulate the gradient and increment the global step.
        ops = [self.policy_loss, self.mse_loss, self.reg_term, self.accuracy ]
        if training:
            ops += [self.grad_op, self.step_op],
        r = self.session.run(ops, feed_dict={self.training: training,
                           self.planes: batch[0],
                           self.probs: batch[1],
                           self.winner: batch[2]})
        # Google's paper scales mse by 1/4 to a [0,1] range, so we do the same here
        return {'policy': r[0], 'mse': r[1]/4., 'reg': r[2],
                'accuracy': r[3], 'total': r[0]+r[1]+r[2] }

    def process(self, train_data, test_data, max_steps=None):
        info_steps=1000
        stats = Stats()
        timer = Timer()
        steps = 0
        while max_steps is None or steps < max_steps:
            try:
                batch = next(train_data)
            except StopIteration:
                print("训练数据已用完，提前结束训练")
                break
            # Measure losses and compute gradients for this batch.
            losses = self.measure_loss(batch, training=True)
            stats.add(losses)
            # fetch the current global step.
            steps = tf.train.global_step(self.session, self.global_step)
            if steps % self.macrobatch == (self.macrobatch-1):
                # Apply the accumulated gradients to the weights.
                self.session.run([self.train_op])
                # Clear the accumulated gradient.
                self.session.run([self.clear_op])

            if steps % info_steps == 0:
                speed = info_steps * self.batch_size / timer.elapsed()
                print("step {}, policy={:g} mse={:g} reg={:g} total={:g} ({:g} pos/s)".format(
                    steps, stats.mean('policy'), stats.mean('mse'), stats.mean('reg'),
                    stats.mean('total'), speed))
                summaries = stats.summaries({'Policy Loss': 'policy',
                                             'MSE Loss': 'mse'})
                self.train_writer.add_summary(
                    tf.Summary(value=summaries), steps)
                stats.clear()

            if steps % 8000 == 0 and steps > 0:
                test_stats = Stats()
                test_batches = 800 # reduce sample mean variance by ~28x
                for _ in range(0, test_batches):
                    try:
                        test_batch = next(test_data)
                    except StopIteration:
                        print("测试数据已用完，跳过剩余测试")
                        break
                    losses = self.measure_loss(test_batch, training=False)
                    test_stats.add(losses)
                summaries = test_stats.summaries({'Policy Loss': 'policy',
                                                  'MSE Loss': 'mse',
                                                  'Accuracy': 'accuracy'})
                self.test_writer.add_summary(tf.Summary(value=summaries), steps)
                print("step {}, policy={:g} training accuracy={:g}%, mse={:g}".\
                    format(steps, test_stats.mean('policy'),
                        test_stats.mean('accuracy')*100.0,
                        test_stats.mean('mse')))

                # Write out current model and checkpoint
                path = os.path.join(os.getcwd(), "leelaz-model")
                save_path = self.saver.save(self.session, path,
                                            global_step=steps)
                print("Model saved in file: {}".format(save_path))
                leela_path = path + "-" + str(steps) + ".txt"
                self.save_leelaz_weights(leela_path)
                print("Leela weights saved to {}".format(leela_path))
                # Things have likely changed enough
                # that stats are no longer valid.

                if self.swa_enabled:
                    self.save_swa_network(steps, path, leela_path, train_data)

                save_path = self.saver.save(self.session, path,
                                            global_step=steps)
                print("Model saved in file: {}".format(save_path))

        # 训练结束，保存最终权重（即使数据提前用完也会执行）
        final_path = os.path.join(os.getcwd(), "leelaz-model-final")
        self.saver.save(self.session, final_path)
        final_leela = final_path + ".txt"
        self.save_leelaz_weights(final_leela)
        print("Final leela weights saved to {}".format(final_leela))

    def save_leelaz_weights(self, filename):
        # Save in PhoenixGo v3 format
        # File layout:
        #   Line 1: "3" (version)
        #   Input conv unit: conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var (6 lines)
        #   Residual blocks × N: 2 units × 6 lines = 12 lines each
        #   Trunk BN: gamma, beta, mean, var (4 lines)
        #   Policy head: conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, ip_w, ip_b (8 lines)
        #   Value head: conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, ip1_w, ip1_b, ip2_w, ip2_b (10 lines)
        with open(filename, "w") as file:
            file.write("3")
            for weights in self.weights:
                file.write("\n")
                work_weights = None
                if weights.name.endswith('/batch_normalization/gamma:0'):
                    # gamma: direct output
                    work_weights = weights
                elif weights.name.endswith('/batch_normalization/beta:0'):
                    # beta: direct output (v3 has explicit beta)
                    work_weights = weights
                elif weights.name.endswith('/batch_normalization/moving_mean:0'):
                    work_weights = weights
                elif weights.name.endswith('/batch_normalization/moving_variance:0'):
                    # Output raw variance (TF stores var, epsilon applied at inference)
                    work_weights = weights
                elif weights.shape.ndims == 4:
                    # Conv: TF [fh,fw,in,out] → Leela [out,in,fh,fw]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # FC: TF [in,out] → Leela [out,in]
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases — direct output
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def reset_batchnorm_key(self):
        self.batch_norm_count = 0
        self.reuse_var = True

    def add_weights(self, var):
        if self.reuse_var is None:
            if var.name[-11:] == "fp16_cast:0":
                name = var.name[:-12] + ":0"
                var = tf.get_default_graph().get_tensor_by_name(name)
            # All trainable variables should be stored as fp32
            assert var.dtype.base_dtype == tf.float32
            self.weights.append(var)

    def batch_norm(self, net):
        # The weights are internal to the batchnorm layer, so apply
        # a unique scope that we can store, and use to look them back up
        # later on.
        scope = self.get_batchnorm_key()
        with tf.variable_scope(scope,
                               custom_getter=float32_variable_storage_getter):
            # PhoenixGo v3: epsilon=1e-3 (not LZ default 1e-5)
            # scale=True (gamma), center=True (beta) — v3 BN has 4 params
            net = tf.layers.batch_normalization(
                    net,
                    epsilon=1e-3, axis=1, fused=True,
                    center=True, scale=True,
                    training=self.training,
                    reuse=self.reuse_var)

        # v3 weight file order: gamma, beta, mean, var
        for v in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
            name = "fp32_storage/" + scope + '/batch_normalization/' + v + ':0'
            var = tf.get_default_graph().get_tensor_by_name(name)
            self.add_weights(var)

        return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels, name):
        # v3 post-activation: conv → +bias → BN → ReLU
        W_conv = weight_variable(
            name,
            [filter_size, filter_size, input_channels, output_channels],
            self.model_dtype)
        b_conv = bias_variable(name + "_b", [output_channels], self.model_dtype)

        self.add_weights(W_conv)
        self.add_weights(b_conv)

        net = inputs
        net = conv2d(net, W_conv)
        net = tf.nn.bias_add(net, b_conv, data_format='NCHW')
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        return net

    def preact_residual_block(self, inputs, channels, name):
        # PhoenixGo v3 pre-activation residual block:
        #   Forward: BN → ReLU → conv+bias → BN → ReLU → conv+bias → +residual (NO ReLU after add)
        #
        # Weight registration order MUST match v3 file order (cw, cb, g, b, m, v)
        # so that save_leelaz_weights outputs in the order Network.cpp load_v3_network reads.
        # Conv weights are created BEFORE batch_norm is applied, but the forward pass
        # still applies BN first (pre-activation). add_weights() order ≠ forward pass order.
        net = inputs
        orig = tf.identity(net)

        # Unit 0: register conv_w, conv_b FIRST (file order), then BN params
        W_conv_1 = weight_variable(name + "_conv_1", [3, 3, channels, channels],
                                   self.model_dtype)
        b_conv_1 = bias_variable(name + "_conv_1_b", [channels], self.model_dtype)
        self.add_weights(W_conv_1)
        self.add_weights(b_conv_1)

        # Forward: pre-activation BN → ReLU → conv
        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        net = conv2d(net, W_conv_1)
        net = tf.nn.bias_add(net, b_conv_1, data_format='NCHW')

        # Unit 1: register conv_w, conv_b FIRST (file order), then BN params
        W_conv_2 = weight_variable(name + "_conv_2", [3, 3, channels, channels],
                                   self.model_dtype)
        b_conv_2 = bias_variable(name + "_conv_2_b", [channels], self.model_dtype)
        self.add_weights(W_conv_2)
        self.add_weights(b_conv_2)

        net = self.batch_norm(net)
        net = tf.nn.relu(net)
        net = conv2d(net, W_conv_2)
        net = tf.nn.bias_add(net, b_conv_2, data_format='NCHW')

        # Add residual (NO ReLU after add — PhoenixGo style)
        net = tf.add(net, orig)
        return net

    def trunk_bn(self, inputs):
        # PhoenixGo trunk BN + ReLU (after residual tower, before heads)
        net = self.batch_norm(inputs)
        net = tf.nn.relu(net)
        return net

    def construct_net(self, planes):
        # NCHW format
        # PhoenixGo v3: 17 channels, 19 x 19
        # Structure (must match Network.cpp load_v3_network + CPUPipe::forward):
        #   1. Input conv (post-act): conv+→bias→BN→ReLU
        #   2. Residual tower (pre-act): BN→ReLU→conv+bias→BN→ReLU→conv+bias→+res (no ReLU)
        #   3. Trunk BN + ReLU (PG-specific, after residual tower)
        #   4. Policy head: conv+bias→BN→ReLU→fc
        #   5. Value head: conv+bias→BN→ReLU→fc→ReLU→fc→tanh
        x_planes = tf.reshape(planes, [-1, 17, 19, 19])

        # 1. Input convolution (post-activation)
        flow = self.conv_block(x_planes, filter_size=3,
                               input_channels=17,
                               output_channels=self.residual_filters,
                               name="first_conv")
        # 2. Residual tower (pre-activation)
        for i in range(0, self.residual_blocks):
            block_name = "res_" + str(i)
            flow = self.preact_residual_block(flow, self.residual_filters,
                                              name=block_name)
        # 3. Trunk BN + ReLU (PhoenixGo-specific)
        flow = self.trunk_bn(flow)

        # 4. Policy head
        conv_pol = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filters,
                                   output_channels=2,
                                   name="policy_head")
        h_conv_pol_flat = tf.reshape(conv_pol, [-1, 2 * 19 * 19])
        W_fc1 = weight_variable("w_fc_1", [2 * 19 * 19, (19 * 19) + 1], self.model_dtype)
        b_fc1 = bias_variable("b_fc_1", [(19 * 19) + 1], self.model_dtype)
        self.add_weights(W_fc1)
        self.add_weights(b_fc1)
        h_fc1 = tf.add(tf.matmul(h_conv_pol_flat, W_fc1), b_fc1)

        # 5. Value head
        conv_val = self.conv_block(flow, filter_size=1,
                                   input_channels=self.residual_filters,
                                   output_channels=1,
                                   name="value_head")
        h_conv_val_flat = tf.reshape(conv_val, [-1, 19 * 19])
        W_fc2 = weight_variable("w_fc_2", [19 * 19, 256], self.model_dtype)
        b_fc2 = bias_variable("b_fc_2", [256], self.model_dtype)
        self.add_weights(W_fc2)
        self.add_weights(b_fc2)
        h_fc2 = tf.nn.relu(tf.add(tf.matmul(h_conv_val_flat, W_fc2), b_fc2))
        W_fc3 = weight_variable("w_fc_3", [256, 1], self.model_dtype)
        b_fc3 = bias_variable("b_fc_3", [1], self.model_dtype)
        self.add_weights(W_fc3)
        self.add_weights(b_fc3)
        h_fc3 = tf.nn.tanh(tf.add(tf.matmul(h_fc2, W_fc3), b_fc3))

        return h_fc1, h_fc3

    def snap_save(self):
        # Save a snapshot of all the variables in the current graph.
        if not hasattr(self, 'save_op'):
            save_ops = []
            rest_ops = []
            for var in self.weights:
                if isinstance(var, str):
                    var = tf.get_default_graph().get_tensor_by_name(var)
                name = var.name.split(':')[0]
                v = tf.Variable(var, name='save/'+name, trainable=False)
                save_ops.append(tf.assign(v, var))
                rest_ops.append(tf.assign(var, v))
            self.save_op = tf.group(*save_ops)
            self.restore_op = tf.group(*rest_ops)
        self.session.run(self.save_op)

    def snap_restore(self):
        # Restore variables in the current graph from the snapshot.
        self.session.run(self.restore_op)

    def save_swa_network(self, steps, path, leela_path, data):
        # Sample 1 in self.swa_c of the networks. Compute in this way so
        # that it's safe to change the value of self.swa_c
        rem = self.session.run(tf.assign_add(self.swa_skip, -1))
        if rem > 0:
            return
        self.swa_skip.load(self.swa_c, self.session)

        # Add the current weight vars to the running average.
        num = self.session.run(self.swa_accum_op)

        if self.swa_max_n != None:
            num = min(num, self.swa_max_n)
            self.swa_count.load(float(num), self.session)

        swa_path = path + "-swa-" + str(int(num)) + "-" + str(steps) + ".txt"

        # save the current network.
        self.snap_save()
        # Copy the swa weights into the current network.
        self.session.run(self.swa_load_op)
        if self.swa_recalc_bn:
            print("Refining SWA batch norm")
            for _ in range(200):
                try:
                    batch = next(data)
                except StopIteration:
                    print("SWA BN recalc: 训练数据已用完，提前结束")
                    break
                self.session.run(
                    [self.loss, self.update_ops],
                    feed_dict={self.training: True,
                               self.planes: batch[0], self.probs: batch[1],
                               self.winner: batch[2]})

        self.save_leelaz_weights(swa_path)
        # restore the saved network.
        self.snap_restore()

        print("Wrote averaged network to {}".format(swa_path))

# Unit tests for TFProcess.
def gen_block(size, f_in, f_out):
    # v3 format: conv_w, conv_b, gamma, beta, mean, var (6 elements per conv+BN unit)
    return [ [1.1] * size * size * f_in * f_out, # conv_w
             [-.1] * f_out,  # conv_b
             [0.5] * f_out,  # gamma
             [0.0] * f_out,  # beta
             [-.2] * f_out,  # moving_mean
             [-.3] * f_out ] # moving_variance

def gen_bn(f_out):
    # trunk BN: gamma, beta, mean, var (4 elements, no conv)
    return [ [0.5] * f_out,  # gamma
             [0.0] * f_out,  # beta
             [-.2] * f_out,  # moving_mean
             [-.3] * f_out ] # moving_variance

class TFProcessTest(unittest.TestCase):
    def test_can_replace_weights(self):
        tfprocess = TFProcess(6, 128)
        tfprocess.init(batch_size=1)
        # use known data to test replace_weights() works.
        data = gen_block(3, 17, tfprocess.residual_filters) # input conv (6)
        for _ in range(tfprocess.residual_blocks):
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))  # unit 0 (6)
            data.extend(gen_block(3,
                tfprocess.residual_filters, tfprocess.residual_filters))  # unit 1 (6)
        # trunk BN (PhoenixGo-specific, 4 lines)
        data.extend(gen_bn(tfprocess.residual_filters))
        # policy head: conv+BN unit (6) + ip_w + ip_b (2) = 8
        data.extend(gen_block(1, tfprocess.residual_filters, 2))
        data.append([0.4] * 2*19*19 * (19*19+1))
        data.append([0.5] * (19*19+1))
        # value head: conv+BN unit (6) + ip1_w + ip1_b + ip2_w + ip2_b (4) = 10
        data.extend(gen_block(1, tfprocess.residual_filters, 1))
        data.append([0.6] * 19*19 * 256)
        data.append([0.7] * 256)
        data.append([0.8] * 256)
        data.append([0.9] * 1)
        tfprocess.replace_weights(data)

if __name__ == '__main__':
    unittest.main()
