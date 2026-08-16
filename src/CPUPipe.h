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

#ifndef CPUPIPE_H_INCLUDED
#define CPUPIPE_H_INCLUDED
#include "config.h"

#include <cassert>
#include <limits>
#include <vector>

#include "ForwardPipe.h"
#include "ThreadPool.h"

class CPUPipe : public ForwardPipe {
public:
    virtual void initialize(int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);
    virtual void forward_batch(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& output_pol,
        std::vector<std::vector<float>>& output_val);
    // CPU batching (USE_BLAS): the eval thread's forward_batch shares the
    // weight reads across the batch and runs the winograd SGEMMs through
    // multi-threaded cblas, while the transforms/elementwise ops are
    // parallelized over a DEDICATED pool (the shared thread pool is occupied
    // by the search threads blocked on their eval results). Without BLAS the
    // per-thread single-forward model is used instead.
    // CPU stays on the per-thread single-forward model: the batched path
    // (dedicated eval pool + parallel transforms) was implemented but its
    // winograd SGEMM measured far slower than expected (~2 n/s vs ~21 n/s
    // per-thread on the 20b net) and needs real profiling to fix. Per-thread
    // forwards remain the best measured configuration.
    virtual bool supports_batching() const override {
        return false;
    }
    // The CPU path allocates per-call buffers, so any batch the user asks
    // for is fine (bounded by memory).
    virtual size_t max_batch_size() const override {
        return std::numeric_limits<size_t>::max();
    }

    virtual void push_weights(
        unsigned int filter_size, unsigned int channels, unsigned int outputs,
        std::shared_ptr<const ForwardPipeWeights> weights);

private:
    void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V, int C);

    void winograd_sgemm(const std::vector<float>& U,
                        const std::vector<float>& V,
                        std::vector<float>& M, int C, int K);

    void winograd_transform_out(const std::vector<float>& M,
                                std::vector<float>& Y, int K);

    void winograd_convolve3(int outputs,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);

    // Batched variants (KataGo-style): multiple positions share one forward
    // pass. The winograd tile columns are indexed as (b*P + p), so the per
    // entry SGEMMs run over `cols = batch*P` columns at once, amortizing the
    // (large) weight reads across the whole batch.
    void winograd_transform_in_batch(const std::vector<float>& in,
                                     std::vector<float>& V, int C, int batch);
    void winograd_sgemm_batch(const std::vector<float>& U,
                              const std::vector<float>& V,
                              std::vector<float>& M, int C, int K, int cols);
    void winograd_transform_out_batch(const std::vector<float>& M,
                                      std::vector<float>& Y, int K, int batch);
    void winograd_convolve3_batch(
        int outputs, const std::vector<float>& input,
        const std::vector<float>& U, int batch, int cols,
        std::vector<float>& V, std::vector<float>& M,
        std::vector<float>& output);

    int m_input_channels;

    // Input + residual block tower
    std::shared_ptr<const ForwardPipeWeights> m_weights;

    std::vector<float> m_conv_pol_w;
    std::vector<float> m_conv_val_w;
    std::vector<float> m_conv_pol_b;
    std::vector<float> m_conv_val_b;

    // Dedicated pool for the batched forward's transforms/elementwise ops
    // (the shared thread pool is held by the search threads waiting on their
    // eval results, so using it here would deadlock).
    Utils::ThreadPool m_evals_pool;
};
#endif
