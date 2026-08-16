/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef FORWARDPIPE_H_INCLUDED
#define FORWARDPIPE_H_INCLUDED

#include "config.h"

#include <memory>
#include <vector>

class ForwardPipe {
public:
    class ForwardPipeWeights {
    public:
        // Input + residual block tower
        std::vector<std::vector<float>> m_conv_weights;
        std::vector<std::vector<float>> m_conv_biases;
        std::vector<std::vector<float>> m_batchnorm_means;
        std::vector<std::vector<float>> m_batchnorm_stddevs;
        // PhoenixGo extension: BN affine params (gamma/beta)
        // v1/v2 networks default to gamma=1.0, beta=0.0 (no affine)
        std::vector<std::vector<float>> m_batchnorm_gammas;
        std::vector<std::vector<float>> m_batchnorm_betas;

        // PhoenixGo extension: trunk BN (after residual tower, before heads)
        // PG network has layer_final/batch_norm; LZ original does not.
        // Empty for v1/v2 networks (no trunk BN).
        std::vector<float> m_bn_trunk_means;
        std::vector<float> m_bn_trunk_stddevs;

        // True when the input convolution has a real (non-identity) batch
        // norm after it. PhoenixGo's 20b-v1 graph is conv+bias with NO input
        // BN (the v3 text file then carries identity gamma/beta/mean/var for
        // the missing BN), while the tfprocess.py / kaggle_go_ai.py training
        // graphs insert a real input BN. The forward pass must apply BN[0]
        // only in the latter case (and the input-conv bias fold in
        // Network::initialize targets BN[0] vs. the residual BNs accordingly).
        bool m_input_has_bn{true};

        // Policy head
        std::vector<float> m_conv_pol_w;
        std::vector<float> m_conv_pol_b;

        std::vector<float> m_conv_val_w;
        std::vector<float> m_conv_val_b;
    };

    virtual ~ForwardPipe() = default;

    virtual void initialize(int channels) = 0;
    virtual bool needs_autodetect() {
        return false;
    };
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;

    // Evaluate a batch of inputs in a single forward pass (KataGo-style
    // batching). The CPU pipe keeps batching off and evaluates each input
    // independently; GPU backends override this to run the whole batch on the
    // device and report supports_batching() == true so Network routes eval
    // requests through its batch queue.
    virtual void forward_batch(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& output_pol,
        std::vector<std::vector<float>>& output_val) = 0;

    virtual bool supports_batching() const {
        return false;
    }
    // Maximum number of positions a single forward_batch() call may take.
    // Backends with fixed-size scratch buffers (OpenCL) return their limit;
    // the eval queue clamps --batchsize to this so a too-large batch cannot
    // overflow the buffers. Pipes that don't batch never get queried.
    virtual size_t max_batch_size() const {
        return 1;
    }
    virtual void push_weights(
        unsigned int filter_size, unsigned int channels, unsigned int outputs,
        std::shared_ptr<const ForwardPipeWeights> weights) = 0;
};

#endif
