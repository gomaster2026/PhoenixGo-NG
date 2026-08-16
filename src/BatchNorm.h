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
*/

#ifndef BATCHNORM_H_INCLUDED
#define BATCHNORM_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <vector>

// Batch normalization fused with ReLU, applied in-place over [channels x
// spatial_size] data laid out as c*spatial_size + b.
//
// `means` and `stddevs` already contain the folded batch-norm parameters
// (scale = gamma / sqrt(var + eps), mean adjusted by beta and any absorbed
// convolution bias — see Network::load_v3_network), so the transform is
// simply y = stddev * (x - mean), then ReLU.
//
// When `eltwise` is given it holds the residual shortcut; the residual add
// happens before the ReLU (no ReLU after the residual add).
template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddevs,
               const float* const eltwise = nullptr) {
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddev = stddevs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = std::max(0.0f, scale_stddev * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] =
                    std::max(0.0f, (scale_stddev * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

#endif
