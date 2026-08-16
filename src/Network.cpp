/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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

#include "config.h"

#include <algorithm>
#include <array>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/utility.hpp>
#include <cassert>
#include <cmath>
#include <exception>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
#include "BatchNorm.h"
#include "CPUPipe.h"
#include "Network.h"
#include "zlib.h"
#ifdef USE_OPENCL
#include "OpenCLScheduler.h"
#include "UCTNode.h"
#endif
#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"

namespace x3 = boost::spirit::x3;
using namespace Utils;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

// Symmetry helper
static std::array<std::array<int, NUM_INTERSECTIONS>, Network::NUM_SYMMETRIES>
    symmetry_nn_idx_table;

float Network::benchmark_time(const int centiseconds) {
    const auto cpus = cfg_num_threads;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    GameState state;
    state.init_game(BOARD_SIZE, KOMI);

    // As a sanity run, try one run with self check.
    // Isn't enough to guarantee correctness but better than nothing,
    // plus for large nets self-check takes a while (1~3 eval per second)
    get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false, true, true);

    const Time start;
    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, start, centiseconds, state]() {
            while (true) {
                runcount++;
                get_output(&state, Ensemble::RANDOM_SYMMETRY, -1, false);
                const Time end;
                const auto elapsed = Time::timediff_centis(start, end);
                if (elapsed >= centiseconds) {
                    break;
                }
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_centis(start, end);
    return 100.0f * runcount.load() / elapsed;
}

void Network::benchmark(const GameState* const state, const int iterations) {
    const auto cpus = cfg_num_threads;
    const Time start;

    ThreadGroup tg(thread_pool);
    std::atomic<int> runcount{0};

    for (auto i = size_t{0}; i < cpus; i++) {
        tg.add_task([this, &runcount, iterations, state]() {
            while (runcount < iterations) {
                runcount++;
                get_output(state, Ensemble::RANDOM_SYMMETRY, -1, false);
            }
        });
    }
    tg.wait_all();

    const Time end;
    const auto elapsed = Time::timediff_seconds(start, end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             runcount.load(), elapsed, int(runcount.load() / elapsed));
}

template <class container>
bool process_bn_var(container& weights, const float epsilon = 1e-5f) {
    // Turn raw variance into 1/sqrt(var + epsilon). Returns false if any value
    // is invalid: negative beyond the epsilon tolerance or non-finite (NaN/Inf)
    // would produce NaN downstream and silently corrupt the network output.
    for (auto&& w : weights) {
        if (!(w + epsilon > 0.0f)) {
            return false;
        }
        w = 1.0f / std::sqrt(w + epsilon);
    }
    return true;
}

// Boost x3's float_ parser accepts "nan"/"inf" tokens, which would inject
// non-finite values straight into the network. Reject any non-finite weight.
template <class container>
bool all_finite(const container& weights) {
    return std::all_of(begin(weights), end(weights),
                       [](const auto w) { return std::isfinite(w); });
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>{
         1.0f,         0.0f,        0.0f,
        -2.0f / 3.0f, -SQ2 / 3.0f, -1.0f / 3.0f,
        -2.0f / 3.0f,  SQ2 / 3.0f, -1.0f / 3.0f,
         1.0f / 6.0f,  SQ2 / 6.0f,  1.0f / 3.0f,
         1.0f / 6.0f, -SQ2 / 6.0f,  1.0f / 3.0f,
         0.0f,         0.0f,        1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs / buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i * 3 + k]
                                   * f[o * channels * 9 + c * 9 + k * 3 + j];
                        }
                        temp[i * 3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi * 3 + k] * G[nu * 3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize
                               + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels + c * outputs + o] =
                        buffer[buffersize * i + entry];
                }
            }
        }
    }

    return U;
}

std::pair<int, int> Network::load_v1_network(std::istream& wtfile) {
    // Count size of the network
    myprintf("Detecting residual layers...");
    // We are version 1 or 2
    if (m_value_head_not_stm) {
        myprintf("v%d...", 2);
    } else {
        myprintf("v%d...", 1);
    }
    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        myprintf("\nInconsistent number of weights in the file.\n");
        return {0, 0};
    }
    residual_blocks /= 8;
    myprintf("%d blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    const auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    // Copies a weight line into a fixed-size array, rejecting size mismatches
    // that would overflow the array or leave it partially uninitialized.
    const auto copy_sized = [](const std::vector<float>& w, auto& dst,
                               const int lineno) {
        if (w.size() != dst.size()) {
            myprintf("\nInvalid weight file: line %d has %d values, expected %d.\n",
                     lineno, int(w.size()), int(dst.size()));
            return false;
        }
        std::copy(cbegin(w), cend(w), begin(dst));
        return true;
    };
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok =
            phrase_parse(it_line, line.cend(), *x3::float_, x3::space, weights);
        // all_finite rejects "nan"/"inf" tokens that boost::spirit would
        // otherwise accept, preventing garbage from reaching the network.
        if (!ok || it_line != line.cend() || !all_finite(weights)) {
            myprintf("\nFailed to parse weight file. Error on line %d.\n",
                     linecount + 2); //+1 from version line, +1 from 0-indexing
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                m_fwd_weights->m_conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                m_fwd_weights->m_conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                m_fwd_weights->m_batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                if (!process_bn_var(weights)) {
                    myprintf("\nInvalid batch-norm variance on line %d.\n",
                             linecount + 2);
                    return {0, 0};
                }
                m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);
            }
        } else {
            switch (linecount - plain_conv_wts) {
                case 0: m_fwd_weights->m_conv_pol_w = std::move(weights); break;
                case 1: m_fwd_weights->m_conv_pol_b = std::move(weights); break;
                case 2:
                    if (!copy_sized(weights, m_bn_pol_w1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 3:
                    if (!copy_sized(weights, m_bn_pol_w2, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 4:
                    if (weights.size()
                        != OUTPUTS_POLICY * NUM_INTERSECTIONS
                               * POTENTIAL_MOVES) {
                        myprintf("The weights file is not for %dx%d boards.\n",
                                 BOARD_SIZE, BOARD_SIZE);
                        return {0, 0};
                    }
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip_pol_w));
                    break;
                case 5:
                    if (!copy_sized(weights, m_ip_pol_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 6: m_fwd_weights->m_conv_val_w = std::move(weights); break;
                case 7: m_fwd_weights->m_conv_val_b = std::move(weights); break;
                case 8:
                    if (!copy_sized(weights, m_bn_val_w1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 9:
                    if (!copy_sized(weights, m_bn_val_w2, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 10:
                    if (!copy_sized(weights, m_ip1_val_w, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 11:
                    if (!copy_sized(weights, m_ip1_val_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 12:
                    if (!copy_sized(weights, m_ip2_val_w, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 13:
                    if (!copy_sized(weights, m_ip2_val_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
            }
        }
        linecount++;
    }
    if (!process_bn_var(m_bn_pol_w2) || !process_bn_var(m_bn_val_w2)) {
        myprintf("\nInvalid batch-norm variance in policy/value head.\n");
        return {0, 0};
    }

    return {channels, static_cast<int>(residual_blocks)};
}

std::pair<int, int> Network::load_v3_network(std::istream& wtfile) {
    // PhoenixGo extension format (v3)
    // Layout per conv+BN unit: conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var (6 lines)
    // Input conv: 1 unit (6 lines) — post-activation conv+BN (applied in CPUPipe::forward)
    // Each residual block: 2 units (12 lines)
    // Trunk BN (layer_final/batch_norm): 4 lines (gamma, beta, mean, var)
    //   — PG-specific, applied after residual tower before policy/value heads
    // Policy head: 8 lines (conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, ip_w, ip_b)
    // Value head: 10 lines (conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, ip1_w, ip1_b, ip2_w, ip2_b)
    myprintf("Detecting residual layers (v3 PhoenixGo format)...");

    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%d channels...", count);
            channels = count;
        }
        linecount++;
    }
    // v3: 1 version + 6 input conv+BN + 12*N residual + 4 trunk BN + 8 policy + 10 value
    constexpr auto TRUNK_BN_LINES = 4;
    auto residual_blocks = linecount - (1 + 6 + TRUNK_BN_LINES + 8 + 10);
    if (residual_blocks % 12 != 0) {
        myprintf("\nInconsistent number of weights in v3 file.\n");
        return {0, 0};
    }
    residual_blocks /= 12;
    myprintf("%d blocks.\n", residual_blocks);

    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way (version line was consumed by
    // load_network_file on first pass, but seekg(0) rewinds past it again).
    std::getline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    const auto plain_conv_wts = plain_conv_layers * 6;  // 6 lines per conv+BN unit
    const auto trunk_bn_start = plain_conv_wts;          // 4 lines trunk BN
    const auto head_start = trunk_bn_start + TRUNK_BN_LINES;
    linecount = 0;
    // Validates a weight line's element count against its expected size.
    const auto check_size = [](const std::vector<float>& w,
                               const size_t expected, const int lineno) {
        if (w.size() != expected) {
            myprintf("\nInvalid weight file: line %d has %d values, expected %d.\n",
                     lineno, int(w.size()), int(expected));
            return false;
        }
        return true;
    };
    // Copies a weight line into a fixed-size array, rejecting size mismatches
    // that would overflow the array or leave it partially uninitialized.
    const auto copy_sized = [](const std::vector<float>& w, auto& dst,
                               const int lineno) {
        if (w.size() != dst.size()) {
            myprintf("\nInvalid weight file: line %d has %d values, expected %d.\n",
                     lineno, int(w.size()), int(dst.size()));
            return false;
        }
        std::copy(cbegin(w), cend(w), begin(dst));
        return true;
    };
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok =
            phrase_parse(it_line, line.cend(), *x3::float_, x3::space, weights);
        // all_finite rejects "nan"/"inf" tokens that boost::spirit would
        // otherwise accept, preventing garbage from reaching the network.
        if (!ok || it_line != line.cend() || !all_finite(weights)) {
            myprintf("\nFailed to parse weight file (line %d contains "
                     "non-numeric or non-finite values).\n",
                     linecount + 2);
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            // 6 lines per conv+BN unit
            const auto line_in_unit = linecount % 6;
            const auto unit_index = linecount / 6;
            if (line_in_unit == 0) {
                // conv_w: [out, in, 3, 3]; the input unit takes INPUT_CHANNELS
                // input planes, all residual units take `channels` in and out.
                const auto expected =
                    unit_index == 0 ? channels * INPUT_CHANNELS * 3 * 3
                                    : channels * channels * 3 * 3;
                if (!check_size(weights, expected, linecount + 2)) {
                    return {0, 0};
                }
                m_fwd_weights->m_conv_weights.emplace_back(
                    std::move(weights));
            } else if (line_in_unit <= 5) {
                // conv_b, bn gamma/beta/mean/var: one value per channel.
                if (!check_size(weights, channels, linecount + 2)) {
                    return {0, 0};
                }
                if (line_in_unit == 1) {
                    m_fwd_weights->m_conv_biases.emplace_back(
                        std::move(weights));
                } else if (line_in_unit == 2) {
                    m_fwd_weights->m_batchnorm_gammas.emplace_back(
                        std::move(weights));
                } else if (line_in_unit == 3) {
                    m_fwd_weights->m_batchnorm_betas.emplace_back(
                        std::move(weights));
                } else if (line_in_unit == 4) {
                    m_fwd_weights->m_batchnorm_means.emplace_back(
                        std::move(weights));
                } else {
                    if (!process_bn_var(weights, 1e-3f)) {
                        myprintf("\nInvalid batch-norm variance on line %d.\n",
                                 linecount + 2);
                        return {0, 0};
                    }
                    m_fwd_weights->m_batchnorm_stddevs.emplace_back(
                        std::move(weights));
                }
            }
        } else if (linecount < head_start) {
            // Trunk BN (4 lines: gamma, beta, mean, var), one per channel.
            if (!check_size(weights, channels, linecount + 2)) {
                return {0, 0};
            }
            switch (linecount - trunk_bn_start) {
                case 0: m_bn_trunk_g1 = std::move(weights); break;
                case 1: m_bn_trunk_b1 = std::move(weights); break;
                case 2: m_bn_trunk_w1 = std::move(weights); break;
                case 3:
                    if (!process_bn_var(weights, 1e-3f)) {
                        myprintf("\nInvalid trunk batch-norm variance on line %d.\n",
                                 linecount + 2);
                        return {0, 0};
                    }
                    m_bn_trunk_w2 = std::move(weights);
                    break;
            }
        } else {
            switch (linecount - head_start) {
                case 0:
                    if (!check_size(weights, OUTPUTS_POLICY * channels,
                                    linecount + 2)) {
                        return {0, 0};
                    }
                    m_fwd_weights->m_conv_pol_w = std::move(weights);
                    break;
                case 1:
                    if (!check_size(weights, OUTPUTS_POLICY, linecount + 2)) {
                        return {0, 0};
                    }
                    m_fwd_weights->m_conv_pol_b = std::move(weights);
                    break;
                case 2:
                    if (!copy_sized(weights, m_bn_pol_g1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 3:
                    if (!copy_sized(weights, m_bn_pol_b1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 4:
                    if (!copy_sized(weights, m_bn_pol_w1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 5:
                    if (!copy_sized(weights, m_bn_pol_w2, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 6:
                    if (weights.size()
                        != OUTPUTS_POLICY * NUM_INTERSECTIONS
                               * POTENTIAL_MOVES) {
                        myprintf("The weights file is not for %dx%d boards.\n",
                                 BOARD_SIZE, BOARD_SIZE);
                        return {0, 0};
                    }
                    std::copy(cbegin(weights), cend(weights),
                              begin(m_ip_pol_w));
                    break;
                case 7:
                    if (!copy_sized(weights, m_ip_pol_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 8:
                    if (!check_size(weights, OUTPUTS_VALUE * channels,
                                    linecount + 2)) {
                        return {0, 0};
                    }
                    m_fwd_weights->m_conv_val_w = std::move(weights);
                    break;
                case 9:
                    if (!check_size(weights, OUTPUTS_VALUE, linecount + 2)) {
                        return {0, 0};
                    }
                    m_fwd_weights->m_conv_val_b = std::move(weights);
                    break;
                case 10:
                    if (!copy_sized(weights, m_bn_val_g1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 11:
                    if (!copy_sized(weights, m_bn_val_b1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 12:
                    if (!copy_sized(weights, m_bn_val_w1, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 13:
                    if (!copy_sized(weights, m_bn_val_w2, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 14:
                    if (!copy_sized(weights, m_ip1_val_w, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 15:
                    if (!copy_sized(weights, m_ip1_val_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 16:
                    if (!copy_sized(weights, m_ip2_val_w, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
                case 17:
                    if (!copy_sized(weights, m_ip2_val_b, linecount + 2)) {
                        return {0, 0};
                    }
                    break;
            }
        }
        linecount++;
    }
    if (!process_bn_var(m_bn_pol_w2, 1e-3f)
        || !process_bn_var(m_bn_val_w2, 1e-3f)) {
        myprintf("\nInvalid batch-norm variance in policy/value head.\n");
        return {0, 0};
    }

    // PhoenixGo: fold gamma/beta into mean/stddev (mathematically equivalent)
    // PG BN: y = gamma*(x-mean)/sqrt(var+eps) + beta
    // LZ BN: y = (x-mean_lz)*stddev_lz, where stddev_lz = 1/sqrt(var+eps)
    // Fold: stddev_lz_new = gamma * stddev_lz
    //       mean_lz_new   = mean - beta / (gamma * stddev_lz) = mean - beta / stddev_lz_new
    // This preserves inference results exactly (mathematical equivalence)
    // Original gamma/beta are also kept in m_batchnorm_gammas/betas for verification
    if (m_format_version == 3) {
        // Residual tower
        for (size_t i = 0; i < m_fwd_weights->m_batchnorm_means.size(); i++) {
            auto& means = m_fwd_weights->m_batchnorm_means[i];
            auto& stddevs = m_fwd_weights->m_batchnorm_stddevs[i];
            const auto& gammas = m_fwd_weights->m_batchnorm_gammas[i];
            const auto& betas = m_fwd_weights->m_batchnorm_betas[i];
            for (size_t j = 0; j < means.size(); j++) {
                if (gammas[j] == 0.0f) {
                    // gamma == 0 would make the folded scale zero and the
                    // mean adjustment infinite, corrupting the whole net.
                    myprintf("\nInvalid weight file: zero batch-norm gamma.\n");
                    return {0, 0};
                }
                const auto new_stddev = gammas[j] * stddevs[j];
                means[j] = means[j] - betas[j] / new_stddev;
                stddevs[j] = new_stddev;
            }
        }
        // Trunk BN (fold gamma/beta into mean/stddev, then copy to ForwardPipeWeights)
        if (!m_bn_trunk_g1.empty()) {
            for (size_t i = 0; i < m_bn_trunk_w1.size(); i++) {
                if (m_bn_trunk_g1[i] == 0.0f) {
                    myprintf("\nInvalid weight file: zero trunk batch-norm gamma.\n");
                    return {0, 0};
                }
                const auto new_stddev = m_bn_trunk_g1[i] * m_bn_trunk_w2[i];
                m_bn_trunk_w1[i] = m_bn_trunk_w1[i] - m_bn_trunk_b1[i] / new_stddev;
                m_bn_trunk_w2[i] = new_stddev;
            }
            m_fwd_weights->m_bn_trunk_means = m_bn_trunk_w1;
            m_fwd_weights->m_bn_trunk_stddevs = m_bn_trunk_w2;
        }
        // Policy head
        for (size_t i = 0; i < m_bn_pol_w1.size(); i++) {
            if (m_bn_pol_g1[i] == 0.0f) {
                myprintf("\nInvalid weight file: zero policy-head batch-norm gamma.\n");
                return {0, 0};
            }
            const auto new_stddev = m_bn_pol_g1[i] * m_bn_pol_w2[i];
            m_bn_pol_w1[i] = m_bn_pol_w1[i] - m_bn_pol_b1[i] / new_stddev;
            m_bn_pol_w2[i] = new_stddev;
        }
        // Value head
        for (size_t i = 0; i < m_bn_val_w1.size(); i++) {
            if (m_bn_val_g1[i] == 0.0f) {
                myprintf("\nInvalid weight file: zero value-head batch-norm gamma.\n");
                return {0, 0};
            }
            const auto new_stddev = m_bn_val_g1[i] * m_bn_val_w2[i];
            m_bn_val_w1[i] = m_bn_val_w1[i] - m_bn_val_b1[i] / new_stddev;
            m_bn_val_w2[i] = new_stddev;
        }
    }

    // Detect whether the input convolution has a real batch norm. The
    // PhoenixGo 20b-v1 graph is conv+bias with NO input BN (the v3 file then
    // carries identity gamma/beta/mean/var for the missing BN), while the
    // tfprocess.py / kaggle_go_ai.py graphs insert a real input BN. The
    // forward pass applies BN[0] (plus ReLU) only when it is non-identity,
    // and the input-conv bias is folded into BN[0] vs. the residual BNs
    // accordingly (see Network::initialize).
    {
        const auto& means = m_fwd_weights->m_batchnorm_means;
        const auto& stddevs = m_fwd_weights->m_batchnorm_stddevs;
        auto input_bn_identity = true;
        if (means.empty() || stddevs.empty()
            || means[0].size() != stddevs[0].size()) {
            input_bn_identity = false;
        } else {
            for (size_t j = 0; j < means[0].size(); j++) {
                if (std::abs(stddevs[0][j] - 1.0f) > 1e-2f
                    || std::abs(means[0][j]) > 1e-3f) {
                    input_bn_identity = false;
                    break;
                }
            }
        }
        m_fwd_weights->m_input_has_bn = !input_bn_identity;
    }

    // The affine gamma/beta parameters have been folded into the means/stddevs
    // above and are no longer needed. Drop them to avoid keeping dead weight.
    if (m_format_version == 3) {
        for (auto& g : m_fwd_weights->m_batchnorm_gammas) {
            std::vector<float>().swap(g);
        }
        for (auto& b : m_fwd_weights->m_batchnorm_betas) {
            std::vector<float>().swap(b);
        }
        m_bn_trunk_g1.clear();
        m_bn_trunk_b1.clear();
        m_bn_trunk_w1.clear();
        m_bn_trunk_w2.clear();
        m_bn_pol_g1.fill(0.0f);
        m_bn_pol_b1.fill(0.0f);
        m_bn_val_g1.fill(0.0f);
        m_bn_val_b1.fill(0.0f);
    }

    return {channels, static_cast<int>(residual_blocks)};
}

std::pair<int, int> Network::load_network_file(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        myprintf("Could not open weights file: %s\n", filename.c_str());
        return {0, 0};
    }
    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            myprintf("Failed to decompress or read: %s\n", filename.c_str());
            gzclose(gzhandle);
            return {0, 0};
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(buffer, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        if (iss.fail()
            || (format_version != 1 && format_version != 2
                && format_version != 3)) {
            myprintf("Weights file is the wrong version.\n");
            return {0, 0};
        } else {
            // Version 2 networks are identical to v1, except
            // that they return the value for black instead of
            // the player to move. This is used by ELF Open Go.
            if (format_version == 2) {
                m_value_head_not_stm = true;
                m_format_version = 2;
            } else if (format_version == 3) {
                // Version 3: PhoenixGo extension format
                // BN has 4 params (gamma/beta/mean/var), 17 input channels
                // value head returns current player's perspective (like v1)
                m_value_head_not_stm = false;
                m_format_version = 3;
            } else {
                m_value_head_not_stm = false;
                m_format_version = 1;
            }
            if (m_format_version == 3) {
                return load_v3_network(buffer);
            }
            // v1/v2 (Leela Zero / ELF) files cannot be evaluated by this
            // engine: the residual tower here is the PhoenixGo pre-activation
            // structure (BN+ReLU before conv), while LZ/ELF v1/v2 networks
            // were trained with post-activation blocks, and the input layout
            // differs (16/18-channel LZ planes vs 17-channel PG planes).
            // Feeding such a file to load_v1_network would misread the input
            // conv as 17 channels — an out-of-bounds read in
            // winograd_transform_f for any real LZ file, or silently wrong
            // output. Reject it explicitly instead.
            myprintf("Unsupported weight format v%d: this engine only supports "
                     "v3 (17-channel PhoenixGo) weight files.\n",
                     format_version);
            return {0, 0};
        }
    }
    return {0, 0};
}

std::unique_ptr<ForwardPipe>&& Network::init_net(
    const int channels, std::unique_ptr<ForwardPipe>&& pipe) {

    pipe->initialize(channels);
    pipe->push_weights(WINOGRAD_ALPHA, INPUT_CHANNELS, channels, m_fwd_weights);

    return std::move(pipe);
}

#ifdef USE_HALF
void Network::select_precision(const int channels) {
    // Only single precision is implemented on the GPU; the HALF path exists
    // for command-line compatibility and computes in float as well.
    myprintf("Initializing OpenCL (single precision).\n");
    m_forward = init_net(channels, std::make_unique<OpenCLScheduler>());
}
#endif

void Network::initialize(const int playouts, const std::string& weightsfile) {
#ifdef USE_BLAS
#ifndef __APPLE__
#ifdef USE_OPENBLAS
    // CPU evals go through the batched eval thread; the winograd SGEMMs are
    // submitted as one cblas_sgemm_batch_strided call per conv, so multi-
    // threaded OpenBLAS finally pays off (per-call overhead used to kill it).
    openblas_set_num_threads(1);
    myprintf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    // mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    myprintf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif
#else
    myprintf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
             EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif

    m_fwd_weights = std::make_shared<ForwardPipeWeights>();

    // Make a guess at a good size as long as the user doesn't
    // explicitly set a maximum memory usage.
    m_nncache.set_size_from_playouts(playouts);

    // Prepare symmetry table
    for (auto s = 0; s < NUM_SYMMETRIES; ++s) {
        for (auto v = 0; v < NUM_INTERSECTIONS; ++v) {
            const auto newvtx =
                get_symmetry({v % BOARD_SIZE, v / BOARD_SIZE}, s);
            symmetry_nn_idx_table[s][v] =
                (newvtx.second * BOARD_SIZE) + newvtx.first;
            assert(symmetry_nn_idx_table[s][v] >= 0
                   && symmetry_nn_idx_table[s][v] < NUM_INTERSECTIONS);
        }
    }

    // Load network from file
    size_t channels, residual_blocks;
    std::tie(channels, residual_blocks) = load_network_file(weightsfile);
    if (channels == 0) {
        exit(EXIT_FAILURE);
    }

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    m_fwd_weights->m_conv_weights[weight_index] = winograd_transform_f(
        m_fwd_weights->m_conv_weights[weight_index], channels, INPUT_CHANNELS);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
        m_fwd_weights->m_conv_weights[weight_index] = winograd_transform_f(
            m_fwd_weights->m_conv_weights[weight_index], channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    if (m_format_version == 3) {
        // PhoenixGo pre-activation residual bias folding (shifted fold).
        //
        // PG block structure (pre-activation):
        //   x_n -> BN_res_n_0 -> ReLU -> conv_res_n_0 -> +bias_res_n_0 -> y_n_0
        //   y_n_0 -> BN_res_n_1 -> ReLU -> conv_res_n_1 -> +bias_res_n_1 -> z_n
        //   x_{n+1} = z_n + x_n  (residual add, no ReLU)
        //
        // Input (post-activation, matches tfprocess.py conv_block):
        //   x_0 = ReLU(BN_input(conv_input(input) + bias_input))
        //
        // To zero out conv biases while preserving equivalence, each bias is
        // absorbed into the mean of the NEXT batch-norm that sees that tensor:
        //   input_conv.bias      -> BN_input.mean (index 0)
        //   bias_res_n_0         -> BN_res_n_1.mean
        //   bias_res_n_1         -> BN_res_{n+1}_0.mean  (or trunk_BN for last)
        //
        // Indexing:
        //   m_conv_biases[0]        = input conv bias
        //   m_conv_biases[1+2n]     = res[n][0].bias
        //   m_conv_biases[2+2n]     = res[n][1].bias
        //   m_batchnorm_means[0]    = input conv BN (applied in CPUPipe::forward)
        //   m_batchnorm_means[1+2n] = res[n][0].bn
        //   m_batchnorm_means[2+2n] = res[n][1].bn
        const auto bias_size = m_fwd_weights->m_conv_biases.size();
        if (bias_size == 0) {
            myprintf("Warning: no conv biases to fold.\n");
        } else {
            const auto channels_per_layer =
                m_fwd_weights->m_conv_biases[0].size();

            // input_conv.bias. Two cases (see m_input_has_bn):
            //  - Input conv has a real BN (tfprocess/GoNet graphs):
            //    conv+bias -> BN[0] -> ReLU. The bias is seen only by BN[0],
            //    fold it into BN[0].mean (index 0).
            //  - No input BN (real PhoenixGo graph): conv+bias feeds straight
            //    into the residual tower. The bias is a constant offset that
            //    propagates through every residual add (x_{n+1} = z_n + x_n,
            //    no ReLU after the add), so it must be folded into each
            //    block's first BN (indices 1,3,5,...) and the trunk BN.
            if (m_fwd_weights->m_input_has_bn) {
                for (size_t j = 0; j < channels_per_layer; j++) {
                    m_fwd_weights->m_batchnorm_means[0][j] -=
                        m_fwd_weights->m_conv_biases[0][j];
                }
            } else {
                for (size_t i = 1; i < m_fwd_weights->m_batchnorm_means.size();
                     i += 2) {
                    for (size_t j = 0; j < channels_per_layer; j++) {
                        m_fwd_weights->m_batchnorm_means[i][j] -=
                            m_fwd_weights->m_conv_biases[0][j];
                    }
                }
                if (!m_fwd_weights->m_bn_trunk_means.empty()) {
                    for (size_t j = 0; j < channels_per_layer; j++) {
                        m_fwd_weights->m_bn_trunk_means[j] -=
                            m_fwd_weights->m_conv_biases[0][j];
                    }
                }
            }
            std::fill(m_fwd_weights->m_conv_biases[0].begin(),
                      m_fwd_weights->m_conv_biases[0].end(), 0.0f);

            // Iterate over residual blocks: i = res[n][0].bias index
            for (size_t i = 1; i < bias_size; i += 2) {
                // bias_res_n_0 -> BN_res_n_1.mean
                for (size_t j = 0; j < channels_per_layer; j++) {
                    m_fwd_weights->m_batchnorm_means[i + 1][j] -=
                        m_fwd_weights->m_conv_biases[i][j];
                }

                // bias_res_n_1 -> next block's BN_res_{n+1}_0.mean,
                // or trunk BN for the last block.
                if (i + 2 < bias_size) {
                    for (size_t j = 0; j < channels_per_layer; j++) {
                        m_fwd_weights->m_batchnorm_means[i + 2][j] -=
                            m_fwd_weights->m_conv_biases[i + 1][j];
                    }
                } else if (!m_fwd_weights->m_bn_trunk_means.empty()) {
                    for (size_t j = 0; j < channels_per_layer; j++) {
                        m_fwd_weights->m_bn_trunk_means[j] -=
                            m_fwd_weights->m_conv_biases[i + 1][j];
                    }
                }

                // Clear conv biases for this block (now folded into BN means)
                std::fill(m_fwd_weights->m_conv_biases[i].begin(),
                          m_fwd_weights->m_conv_biases[i].end(), 0.0f);
                std::fill(m_fwd_weights->m_conv_biases[i + 1].begin(),
                          m_fwd_weights->m_conv_biases[i + 1].end(), 0.0f);
            }
        }
    } else {
        // v1/v2 standard bias folding (post-activation: conv[i].bias -> BN[i].mean)
        auto bias_size = m_fwd_weights->m_conv_biases.size();
        for (auto i = size_t{0}; i < bias_size; i++) {
            auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
            for (auto j = size_t{0}; j < means_size; j++) {
                m_fwd_weights->m_batchnorm_means[i][j] -=
                    m_fwd_weights->m_conv_biases[i][j];
                m_fwd_weights->m_conv_biases[i][j] = 0.0f;
            }
        }
    }

    // Heads (post-activation for all versions): conv bias -> same BN mean
    for (auto i = size_t{0}; i < m_bn_val_w1.size(); i++) {
        m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
        m_fwd_weights->m_conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_bn_pol_w1.size(); i++) {
        m_bn_pol_w1[i] -= m_fwd_weights->m_conv_pol_b[i];
        m_fwd_weights->m_conv_pol_b[i] = 0.0f;
    }

#ifdef USE_OPENCL
    if (cfg_cpu_only) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward = init_net(channels, std::make_unique<CPUPipe>());
    } else {
#ifdef USE_OPENCL_SELFCHECK
        // initialize CPU reference first, so that we can self-check
        // when doing fp16 vs. fp32 detections
        m_forward_cpu = init_net(channels, std::make_unique<CPUPipe>());
#endif
#ifdef USE_HALF
        // HALF support is enabled, and we are using the GPU.
        select_precision(channels);
#else
        myprintf("Initializing OpenCL (single precision).\n");
        m_forward = init_net(channels, std::make_unique<OpenCLScheduler>());
#endif
    }

#else // !USE_OPENCL
    myprintf("Initializing CPU-only evaluation.\n");
    m_forward = init_net(channels, std::make_unique<CPUPipe>());
#endif

    // Need to estimate size before clearing up the pipe.
    get_estimated_size();
    m_fwd_weights.reset();

    if (m_forward->supports_batching()) {
        // Start the batch eval thread (batched GPU backend). Honor the
        // engine's --batchsize option, clamped to what the backend can
        // evaluate in a single forward pass (the OpenCL scratch buffers are
        // sized for OpenCL::MAX_BATCH positions; a larger batch would
        // overflow them).
        m_eval_batch_size = std::min<size_t>(std::max(cfg_batch_size, 1u),
                                             m_forward->max_batch_size());
        m_eval_running = true;
        m_eval_thread = std::thread([this] { eval_thread_loop(); });
    }
}

template <unsigned int inputs, unsigned int outputs, bool ReLU, size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases) {
    std::vector<float> output(outputs);

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(), inputs, outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = std::max(0.0f, val);
        }
        output[o] = val;
    }

    return output;
}

#ifdef USE_OPENCL_SELFCHECK
void Network::compare_net_outputs(const Netresult& data, const Netresult& ref) {
    // Calculates L2-norm between data and ref.
    constexpr auto max_error = 0.2f;

    auto error = 0.0f;

    for (auto idx = size_t{0}; idx < data.policy.size(); ++idx) {
        const auto diff = data.policy[idx] - ref.policy[idx];
        error += diff * diff;
    }
    const auto diff_pass = data.policy_pass - ref.policy_pass;
    const auto diff_winrate = data.winrate - ref.winrate;
    error += diff_pass * diff_pass;
    error += diff_winrate * diff_winrate;

    error = std::sqrt(error);

    if (error > max_error || std::isnan(error)) {
        printf(
            "Error in OpenCL calculation: Update your device's OpenCL drivers "
            "or reduce the amount of games played simultaneously.\n");
        throw std::runtime_error("OpenCL self-check mismatch.");
    }
}
#endif

std::vector<float> softmax(const std::vector<float>& input,
                           const float temperature = 1.0f) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

Network::~Network() {
    if (m_eval_running.exchange(false)) {
        m_eval_cv.notify_all();
        if (m_eval_thread.joinable()) {
            m_eval_thread.join();
        }
    }
}

// Runs one forward pass for a batch of evaluation requests gathered from the
// search threads (KataGo-style batching).
void Network::eval_thread_loop() {
    while (m_eval_running) {
        std::vector<EvalRequest> batch;
        {
            std::unique_lock<std::mutex> lock(m_eval_mutex);
            m_eval_cv.wait(lock, [this] {
                return !m_eval_queue.empty() || !m_eval_running;
            });
            if (!m_eval_running && m_eval_queue.empty()) {
                return;
            }
            while (!m_eval_queue.empty()
                   && batch.size() < m_eval_batch_size) {
                batch.push_back(std::move(m_eval_queue.front()));
                m_eval_queue.pop_front();
            }
            // KataGo-style grace period: when requests arrive one at a time
            // (self-play / benchmark), serve them as one batch instead of
            // running a single-position forward per request. Without this the
            // weight-read amortization that makes batching worthwhile is lost,
            // and CPU batching is barely faster than no batching at all.
            if (batch.size() < m_eval_batch_size && !batch.empty()) {
                m_eval_cv.wait_for(lock, std::chrono::microseconds(1000),
                                   [this] { return !m_eval_queue.empty(); });
                while (!m_eval_queue.empty()
                       && batch.size() < m_eval_batch_size) {
                    batch.push_back(std::move(m_eval_queue.front()));
                    m_eval_queue.pop_front();
                }
            }
        }
        if (batch.empty()) {
            continue;
        }
        std::vector<const std::vector<float>*> inputs;
        inputs.reserve(batch.size());
        for (const auto& request : batch) {
            inputs.push_back(&request.input);
        }
        std::vector<std::vector<float>> pols(batch.size());
        std::vector<std::vector<float>> vals(batch.size());
        try {
            m_forward->forward_batch(inputs, pols, vals);
        } catch (const std::exception& e) {
            // A failed forward (e.g. GPU device lost) must not leave the
            // waiting search threads blocked forever, nor let the exception
            // escape this thread and abort the process via std::terminate.
            // Deliver correctly-sized zero results (uniform policy, ~0.5
            // value) so the search degrades gracefully instead of crashing.
            myprintf("Network eval failed: %s\n", e.what());
            const auto pol_size =
                Network::OUTPUTS_POLICY * NUM_INTERSECTIONS;
            const auto val_size =
                Network::OUTPUTS_VALUE * NUM_INTERSECTIONS;
            for (auto& request : batch) {
                request.result.set_value(
                    {std::vector<float>(pol_size, 0.0f),
                     std::vector<float>(val_size, 0.0f)});
            }
            continue;
        }
        for (size_t i = 0; i < batch.size(); i++) {
            batch[i].result.set_value(
                {std::move(pols[i]), std::move(vals[i])});
        }
    }
}

// Submit a single evaluation through the batch queue and wait for the result.
// Only used when the active pipe reports supports_batching().
void Network::forward_queued(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val) {
    if (m_eval_drain.load()) {
        // Search is being shut down (ponder interrupted, move received):
        // bail out instead of submitting an eval that would be discarded.
        throw NetworkHaltException{};
    }
    assert(m_eval_running.load());
    std::promise<std::pair<std::vector<float>, std::vector<float>>> promise;
    auto future = promise.get_future();
    {
        std::lock_guard<std::mutex> lock(m_eval_mutex);
        m_eval_queue.emplace_back(EvalRequest{input, std::move(promise)});
    }
    m_eval_cv.notify_one();
    auto result = future.get();
    output_pol = std::move(result.first);
    output_val = std::move(result.second);
}

bool Network::probe_cache(const GameState* const state,
                          Network::Netresult& result) {
    if (m_nncache.lookup(state->board.get_hash(), result)) {
        return true;
    }
    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cfg_noise && !cfg_random_cnt
        && state->get_movenum()
               < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
            if (sym == Network::IDENTITY_SYMMETRY) {
                continue;
            }
            const auto hash = state->get_symmetry_hash(sym);
            if (m_nncache.lookup(hash, result)) {
                decltype(result.policy) corrected_policy;
                for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
                    const auto sym_idx = symmetry_nn_idx_table[sym][idx];
                    corrected_policy[idx] = result.policy[sym_idx];
                }
                result.policy = std::move(corrected_policy);
                return true;
            }
        }
    }

    return false;
}

Network::Netresult Network::get_output(
    const GameState* const state, const Ensemble ensemble, const int symmetry,
    const bool read_cache, const bool write_cache, const bool force_selfcheck) {
    Netresult result;
    if (state->board.get_boardsize() != BOARD_SIZE) {
        return result;
    }

    if (read_cache) {
        // See if we already have this in the cache.
        if (probe_cache(state, result)) {
            return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else if (ensemble == AVERAGE) {
        assert(symmetry == -1);
        for (auto sym = 0; sym < NUM_SYMMETRIES; ++sym) {
            auto tmpresult = get_output_internal(state, sym);
            result.winrate +=
                tmpresult.winrate / static_cast<float>(NUM_SYMMETRIES);
            result.policy_pass +=
                tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

            for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
                result.policy[idx] +=
                    tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
            }
        }
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
#ifdef USE_OPENCL_SELFCHECK
        // Both implementations are available, self-check the OpenCL driver by
        // running both with a probability of 1/2000.
        // selfcheck is done here because this is the only place NN
        // evaluation is done on actual gameplay.
        if (m_forward_cpu != nullptr
            && (force_selfcheck
                || Random::get_Rng().randfix<SELFCHECK_PROBABILITY>() == 0)) {
            auto result_ref = get_output_internal(state, rand_sym, true);
            compare_net_outputs(result, result_ref);
        }
#else
        (void)force_selfcheck;
#endif
    }

    // v2 format (ELF Open Go) returns black value, not stm
    if (m_value_head_not_stm) {
        if (state->board.get_to_move() == FastBoard::WHITE) {
            result.winrate = 1.0f - result.winrate;
        }
    }

    if (write_cache) {
        // Insert result into cache.
        m_nncache.insert(state->board.get_hash(), result);
    }

    return result;
}

Network::Netresult Network::get_output_internal(const GameState* const state,
                                                const int symmetry,
                                                bool selfcheck) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    constexpr auto width = BOARD_SIZE;
    constexpr auto height = BOARD_SIZE;

    const auto input_data = gather_features(state, symmetry);
    std::vector<float> policy_data(OUTPUTS_POLICY * width * height);
    std::vector<float> value_data(OUTPUTS_VALUE * width * height);
#ifdef USE_OPENCL_SELFCHECK
    if (selfcheck) {
        m_forward_cpu->forward(input_data, policy_data, value_data);
    } else if (m_forward->supports_batching()) {
        forward_queued(input_data, policy_data, value_data);
    } else {
        m_forward->forward(input_data, policy_data, value_data);
    }
#else
    if (m_forward->supports_batching()) {
        forward_queued(input_data, policy_data, value_data);
    } else {
        m_forward->forward(input_data, policy_data, value_data);
    }
    (void)selfcheck;
#endif

    // Get the moves
    batchnorm<NUM_INTERSECTIONS>(OUTPUTS_POLICY, policy_data,
                                 m_bn_pol_w1.data(), m_bn_pol_w2.data());

    // PhoenixGo (v3) policy dense layer expects NHWC flatten layout
    // (y*W*Cin + x*Cin + c), but convolve<1> outputs CHW (c*H*W + y*W + x).
    // Convert CHW -> NHWC for v3 networks. v1/v2 keep CHW (LZ native).
    std::vector<float> policy_data_dense;
    if (m_format_version == 3) {
        policy_data_dense.resize(OUTPUTS_POLICY * NUM_INTERSECTIONS);
        for (auto c = 0; c < OUTPUTS_POLICY; c++) {
            for (auto y = 0; y < BOARD_SIZE; y++) {
                for (auto x = 0; x < BOARD_SIZE; x++) {
                    const auto chw_idx =
                        c * NUM_INTERSECTIONS + y * BOARD_SIZE + x;
                    const auto nhwc_idx =
                        y * BOARD_SIZE * OUTPUTS_POLICY + x * OUTPUTS_POLICY + c;
                    policy_data_dense[nhwc_idx] = policy_data[chw_idx];
                }
            }
        }
    } else {
        policy_data_dense = policy_data;
    }
    const auto policy_out =
        innerproduct<OUTPUTS_POLICY * NUM_INTERSECTIONS, POTENTIAL_MOVES,
                     false>(policy_data_dense, m_ip_pol_w, m_ip_pol_b);
    const auto outputs = softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    batchnorm<NUM_INTERSECTIONS>(OUTPUTS_VALUE, value_data, m_bn_val_w1.data(),
                                 m_bn_val_w2.data());
    const auto winrate_data =
        innerproduct<OUTPUTS_VALUE * NUM_INTERSECTIONS, VALUE_LAYER, true>(
            value_data, m_ip1_val_w, m_ip1_val_b);
    const auto winrate_out = innerproduct<VALUE_LAYER, 1, false>(
        winrate_data, m_ip2_val_w, m_ip2_val_b);

    // LZ winrate = (1 + tanh(value_head_output)) / 2
    // PhoenixGo's value_tensor (raw tanh output from network) uses standard
    // convention: +1 = current player wins, -1 = loses. PG wraps it as
    // value = -value_tensor for its own MCTS (see zero_model.cc:169 and
    // mcts_engine.cc:302), but that is PG's internal convention.
    // LZ should match the RAW network output, which is identical to LZ's
    // original convention -- so the original (1+tanh)/2 formula is correct.
    const auto winrate = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

    Netresult result;

    for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = outputs[idx];
    }

    result.policy_pass = outputs[NUM_INTERSECTIONS];
    result.winrate = winrate;

    return result;
}

void Network::show_heatmap(const FastState* const state,
                           const Netresult& result, const bool topmoves) {
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < BOARD_SIZE; y++) {
        for (unsigned int x = 0; x < BOARD_SIZE; x++) {
            auto policy = 0;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                policy = result.policy[y * BOARD_SIZE + x] * 1000;
            }

            line += boost::str(boost::format("%3d ") % policy);
        }

        display_map.push_back(line);
        line.clear();
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    const auto pass_policy = int(result.policy_pass * 1000);
    myprintf("pass: %d\n", pass_policy);
    myprintf("winrate: %f\n", result.winrate);

    if (topmoves) {
        std::vector<Network::PolicyVertexPair> moves;
        for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
            const auto x = i % BOARD_SIZE;
            const auto y = i / BOARD_SIZE;
            const auto vertex = state->board.get_vertex(x, y);
            if (state->board.get_state(vertex) == FastBoard::EMPTY) {
                moves.emplace_back(result.policy[i], vertex);
            }
        }
        moves.emplace_back(result.policy_pass, FastBoard::PASS);

        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        for (const auto& move : moves) {
            if (cum > 0.85f || move.first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                     move.first,
                     state->board.move_to_text(move.second).c_str());
            cum += move.first;
        }
    }
}

void Network::fill_input_plane_pair(const FullBoard& board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
    for (auto idx = 0; idx < NUM_INTERSECTIONS; idx++) {
        const auto sym_idx = symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % BOARD_SIZE;
        const auto y = sym_idx / BOARD_SIZE;
        const auto color = board.get_state(x, y);
        if (color == FastBoard::BLACK) {
            black[idx] = float(true);
        } else if (color == FastBoard::WHITE) {
            white[idx] = float(true);
        }
    }
}

std::vector<float> Network::gather_features(const GameState* const state,
                                            const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    // PhoenixGo: 17 channels = 8 history pairs + 1 color
    // Layout (interleaved): me_t, opp_t, me_{t-1}, opp_{t-1}, ..., color
    // See PhoenixGo common/go_state.cc:225-245 GetFeature
    auto input_data = std::vector<float>(INPUT_CHANNELS * NUM_INTERSECTIONS);

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    // PhoenixGo reverse_plane mechanism: when black to move, swap each
    // history pair so "current player" is always first (me/opp perspective)
    // LZ original uses black_it/white_it pointer swap for same effect
    const auto me_it =
        blacks_move ? begin(input_data)
                    : begin(input_data) + NUM_INTERSECTIONS;
    const auto opp_it =
        blacks_move ? begin(input_data) + NUM_INTERSECTIONS
                    : begin(input_data);

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards (interleaved: pair 0,1 at t,
    // pair 2,3 at t-1, ...)
    for (auto h = size_t{0}; h < moves; h++) {
        // Each history pair occupies 2 channels, interleaved layout
        const auto pair_offset = 2 * h * NUM_INTERSECTIONS;
        fill_input_plane_pair(state->get_past_board(h),
                              me_it + pair_offset,
                              opp_it + pair_offset, symmetry);
    }

    // PhoenixGo: single color channel (channel 16)
    // Black to move = 1.0, White to move = 0.0
    // (PG go_state.cc:238-242 only fills 1 when Self()==BLACK)
    const auto color_it =
        begin(input_data) + 2 * INPUT_MOVES * NUM_INTERSECTIONS;
    if (blacks_move) {
        std::fill(color_it, color_it + NUM_INTERSECTIONS, float(true));
    } else {
        std::fill(color_it, color_it + NUM_INTERSECTIONS, float(false));
    }

    return input_data;
}

std::pair<int, int> Network::get_symmetry(const std::pair<int, int>& vertex,
                                          const int symmetry,
                                          const int board_size) {
    auto x = vertex.first;
    auto y = vertex.second;
    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    if ((symmetry & 4) != 0) {
        std::swap(x, y);
    }

    if ((symmetry & 2) != 0) {
        x = board_size - x - 1;
    }

    if ((symmetry & 1) != 0) {
        y = board_size - y - 1;
    }

    assert(x >= 0 && x < board_size);
    assert(y >= 0 && y < board_size);
    assert(symmetry != IDENTITY_SYMMETRY || vertex == std::make_pair(x, y));
    return {x, y};
}

size_t Network::get_estimated_size() {
    if (estimated_size != 0) {
        return estimated_size;
    }
    auto result = size_t{0};

    const auto lambda_vector_size =
        [](const std::vector<std::vector<float>>& v) {
            auto result = size_t{0};
            for (auto it = begin(v); it != end(v); ++it) {
                result += it->size() * sizeof(float);
            }
            return result;
        };

    result += lambda_vector_size(m_fwd_weights->m_conv_weights);
    result += lambda_vector_size(m_fwd_weights->m_conv_biases);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_means);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_stddevs);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_gammas);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_betas);
    result += m_fwd_weights->m_bn_trunk_means.size() * sizeof(float);
    result += m_fwd_weights->m_bn_trunk_stddevs.size() * sizeof(float);

    result += m_fwd_weights->m_conv_pol_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_pol_b.size() * sizeof(float);

    // Policy head
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_w1
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_w2
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_g1
    result += OUTPUTS_POLICY * sizeof(float); // m_bn_pol_b1
    result += OUTPUTS_POLICY * NUM_INTERSECTIONS * POTENTIAL_MOVES
              * sizeof(float);                 // m_ip_pol_w
    result += POTENTIAL_MOVES * sizeof(float); // m_ip_pol_b

    // Value head
    result += m_fwd_weights->m_conv_val_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_val_b.size() * sizeof(float);
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_w1
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_w2
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_g1
    result += OUTPUTS_VALUE * sizeof(float); // m_bn_val_b1

    result += OUTPUTS_VALUE * NUM_INTERSECTIONS * VALUE_LAYER
              * sizeof(float);             // m_ip1_val_w
    result += VALUE_LAYER * sizeof(float); // m_ip1_val_b

    result += VALUE_LAYER * sizeof(float); // m_ip2_val_w
    result += sizeof(float);               // m_ip2_val_b
    return estimated_size = result;
}

size_t Network::get_estimated_cache_size() {
    return m_nncache.get_estimated_size();
}

void Network::nncache_resize(const int max_count) {
    return m_nncache.resize(max_count);
}

void Network::nncache_clear() {
    m_nncache.clear();
}

void Network::drain_evals() {
    // Stop new eval submissions (they throw NetworkHaltException so search
    // threads bail out promptly). Requests already in the queue are still
    // served by the eval thread, so threads blocked on them cannot hang.
    m_eval_drain.store(true);
}

void Network::resume_evals() {
    m_eval_drain.store(false);
}
