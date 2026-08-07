"""
Mixed precision training utilities for TensorFlow.
Supports FP16/FP32 automatic selection.
"""

import tensorflow as tf
import numpy as np

def get_policy_and_value_batch(data):
    """
    Parse a batch of training data.
    Returns: (planes, policy, value)
    """
    batch_size = len(data)
    planes = np.zeros((batch_size, 18, 19, 19), dtype=np.float32)
    policy = np.zeros((batch_size, 362), dtype=np.float32)
    value = np.zeros((batch_size, 1), dtype=np.float32)
    
    for i, record in enumerate(data):
        if len(record) < 19:
            continue
        # Parse first 16 planes from hex
        for plane_idx in range(16):
            hex_str = record[plane_idx]
            bits = bin(int(hex_str, 16))[2:].zfill(361)
            for bit_idx, bit in enumerate(bits):
                if bit == '1':
                    y = bit_idx // 19
                    x = bit_idx % 19
                    planes[i, plane_idx, y, x] = 1.0
        # Player to move
        player = int(record[16])
        planes[i, 16, :, :] = player
        planes[i, 17, :, :] = 1 - player
        # Policy (MCTS probabilities)
        policy_values = [float(x) for x in record[17].split()]
        policy[i, :] = policy_values[:362]
        # Value (win/loss)
        value[i, 0] = float(record[18])
    
    return planes, policy, value

def get_batch_format(data):
    """Get mixed precision batch format for training."""
    planes, policy, value = get_policy_and_value_batch(data)
    return {
        'planes': tf.constant(planes, dtype=tf.float32),
        'policy': tf.constant(policy, dtype=tf.float32),
        'value': tf.constant(value, dtype=tf.float32),
    }