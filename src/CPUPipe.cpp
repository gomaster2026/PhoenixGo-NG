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

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif
// The winograd SGEMMs and head convs always use Eigen (cblas is measurably
// slower on these small matrices and does not scale across search threads),
// so the Eigen headers/helpers are needed in every build.
#include <Eigen/Dense>

#include "BatchNorm.h"
#include "CPUPipe.h"
#include "GTP.h"
#include "Im2Col.h"
#include "Network.h"
#include "ThreadPool.h"
#include "Utils.h"

// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

void CPUPipe::initialize(int channels) {
    m_input_channels = channels;
#ifdef USE_BLAS
    // Dedicated eval-side pool for the batched transforms (see header). Only
    // needed when the batched path is active.
    m_evals_pool.initialize(std::max(1u, cfg_num_threads));
#endif
}

void CPUPipe::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V, const int C) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    constexpr auto buffersize = 32;

    std::array<std::array<float, Wpad>, Wpad> in_pad{{{0.0f}}};

    std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
    auto buffer_offset = 0;
    auto buffer_entries = 0;

    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, WINOGRAD_TILE>{
    //     1.0f,  0.0f,       -5.0f / 2.0f,  0.0f,        1.0f, 0.0f,
    //     0.0f, -SQ2,        -2.0f,         SQ2 / 2.0f,  1.0f, 0.0f,
    //     0.0f,  SQ2,        -2.0f,        -SQ2 / 2.0f,  1.0f, 0.0f,
    //     0.0f, -SQ2 / 2.0f, -1.0f / 2.0f,  SQ2,         1.0f, 0.0f,
    //     0.0f,  SQ2 / 2.0f, -1.0f / 2.0f, -SQ2,         1.0f, 0.0f,
    //     0.0f,  1.0f,        0.0f,        -5.0f / 2.0f, 0.0f, 1.0f};
    const auto multiply_bt = [](float& o0, float& o1, float& o2,
                                float& o3, float& o4, float& o5,
                                const float i0, const float i1, const float i2,
                                const float i3, const float i4, const float i5) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
        auto i4m2_2 = i2 * (-1.0f / 2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
    };

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch * (W * H) + yin * W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX)                                                            \
    float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,     \
        T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

                // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX)                                                        \
    multiply_bt(T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX,         \
                T1_5_##XX,                                                     \
                in_pad[yin + 0][xin + XX],                                     \
                in_pad[yin + 1][xin + XX],                                     \
                in_pad[yin + 2][xin + XX],                                     \
                in_pad[yin + 3][xin + XX],                                     \
                in_pad[yin + 4][xin + XX],                                     \
                in_pad[yin + 5][xin + XX]);
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX)                                                         \
    multiply_bt(                                                               \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries],       \
        T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,       \
        T1_##XX##_5);
                MULTIPLY_B(0)
                MULTIPLY_B(1)
                MULTIPLY_B(2)
                MULTIPLY_B(3)
                MULTIPLY_B(4)
                MULTIPLY_B(5)

                if (buffer_entries == 0) {
                    buffer_offset = ch * P + block_y * WTILES + block_x;
                }
                buffer_entries++;

                if (buffer_entries >= buffersize
                    || (ch == C - 1 && block_x == WTILES - 1
                        && block_y == WTILES - 1)) {

                    for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                        for (auto entry = 0; entry < buffer_entries; entry++) {
                            V[i * C * P + buffer_offset + entry] =
                                buffer[i * buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

void CPUPipe::winograd_sgemm(const std::vector<float>& U,
                             const std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    // Per-thread single-position path (search threads each run their own
    // forward). Eigen is faster than cblas for these tiny winograd GEMMs
    // (P=25 result columns; cblas pays per-call overhead on 36 small GEMMs
    // per conv). Measured on the 20b net: Eigen per-thread ~22 n/s total on
    // 6 threads vs ~7 n/s with cblas.
    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;
        auto C_mat = EigenMatrixMap<float>(M.data() + offset_m, P, K);
        C_mat.noalias() =
            ConstEigenMatrixMap<float>(V.data() + offset_v, P, C)
            * ConstEigenMatrixMap<float>(U.data() + offset_u, K, C).transpose();
    }
}

void CPUPipe::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y, const int K) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>{
    //     1.0f, 1.0f,        1.0f,        1.0f,        1.0f,       0.0f,
    //     0.0f, SQ2 / 2.0f, -SQ2 / 2.0f,  SQ2,        -SQ2,        0.0f,
    //     0.0f, 1.0f / 2.0f, 1.0f / 2.0f, 2.0f,        2.0f,       0.0f,
    //     0.0f, SQ2 / 4.0f, -SQ2 / 4.0f,  2.0f * SQ2, -2.0f * SQ2, 1.0f};
    const auto multiply_at = [](float& o0, float& o1, float& o2, float& o3,
                                const float i0, const float i1,
                                const float i2, const float i3,
                                const float i4, const float i5) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2 / 4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);

        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < WTILES; block_x++) {
            const auto x = WINOGRAD_M * block_x;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto y = WINOGRAD_M * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>,
                               WINOGRAD_ALPHA>;
                WinogradTile temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[(xi * WINOGRAD_ALPHA + nu) * K * P + k * P + b];
                    }
                }
                std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M> temp;
                std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M> o;

                // Calculates transpose(A).temp_m.A
                for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                    multiply_at(temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                                temp_m[0][j], temp_m[1][j], temp_m[2][j],
                                temp_m[3][j], temp_m[4][j], temp_m[5][j]);
                }

                for (auto i = 0; i < WINOGRAD_M; i++) {
                    multiply_at(o[i][0], o[i][1], o[i][2], o[i][3],
                                temp[i][0], temp[i][1], temp[i][2],
                                temp[i][3], temp[i][4], temp[i][5]);
                }

                const auto y_ind = k * H * W + y * W + x;
                for (auto i = 0; i < WINOGRAD_M; i++) {
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

void CPUPipe::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

// Batched (KataGo-style) winograd: same transforms as the single-position
// versions, but the tile columns are indexed as (b*P + p), so one SGEMM per
// transform entry covers the whole batch. Input is [batch][C][19*19], output
// is [batch][K][19*19], V/M are [36][C/K][batch*P].
void CPUPipe::winograd_transform_in_batch(const std::vector<float>& in,
                                          std::vector<float>& V, const int C,
                                          const int batch) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;
    const auto cols = batch * P;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    const auto multiply_bt = [](float& o0, float& o1, float& o2,
                                float& o3, float& o4, float& o5,
                                const float i0, const float i1, const float i2,
                                const float i3, const float i4, const float i5) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;
        o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;
        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
        auto i4m2_2 = i2 * (-1.0f / 2.0f) + i4;
        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;
        o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
    };

    // Each (b, ch) pair writes its own consecutive V slots, so the transform
    // parallelizes over (b*C+ch) ranges. Every task has its own in_pad/buffer
    // (the padded border is zero once; the 19x19 interior is refilled per
    // channel, exactly as the sequential version did).
    const auto process_range = [&](const size_t start, const size_t end) {
        constexpr auto buffersize = 32;
        std::array<std::array<float, Wpad>, Wpad> in_pad{{{0.0f}}};
        std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
        for (auto idx = start; idx < end; idx++) {
            const auto b = static_cast<int>(idx / C);
            const auto ch = static_cast<int>(idx % C);
            for (auto yin = 0; yin < H; yin++) {
                for (auto xin = 0; xin < W; xin++) {
                    in_pad[yin + 1][xin + 1] =
                        in[(b * C + ch) * (W * H) + yin * W + xin];
                }
            }
            auto buffer_offset = 0;
            auto buffer_entries = 0;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto yin = WINOGRAD_M * block_y;
                for (auto block_x = 0; block_x < WTILES; block_x++) {
                    const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX)                                                            \
    float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,     \
        T1_##XX##_5;
                    DECL_T1(0)
                    DECL_T1(1)
                    DECL_T1(2)
                    DECL_T1(3)
                    DECL_T1(4)
                    DECL_T1(5)

#define MULTIPLY_BT(XX)                                                        \
    multiply_bt(T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX,         \
                T1_5_##XX,                                                     \
                in_pad[yin + 0][xin + XX],                                     \
                in_pad[yin + 1][xin + XX],                                     \
                in_pad[yin + 2][xin + XX],                                     \
                in_pad[yin + 3][xin + XX],                                     \
                in_pad[yin + 4][xin + XX],                                     \
                in_pad[yin + 5][xin + XX]);
                    MULTIPLY_BT(0)
                    MULTIPLY_BT(1)
                    MULTIPLY_BT(2)
                    MULTIPLY_BT(3)
                    MULTIPLY_BT(4)
                    MULTIPLY_BT(5)

#define MULTIPLY_B(XX)                                                         \
    multiply_bt(                                                               \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries],       \
        buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries],       \
        T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,       \
        T1_##XX##_5);
                    MULTIPLY_B(0)
                    MULTIPLY_B(1)
                    MULTIPLY_B(2)
                    MULTIPLY_B(3)
                    MULTIPLY_B(4)
                    MULTIPLY_B(5)
#undef DECL_T1
#undef MULTIPLY_BT
#undef MULTIPLY_B

                    if (buffer_entries == 0) {
                        buffer_offset =
                            ch * cols + b * P + block_y * WTILES + block_x;
                    }
                    buffer_entries++;

                    if (block_y == WTILES - 1 && block_x == WTILES - 1) {
                        for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA;
                             i++) {
                            for (auto entry = 0; entry < buffer_entries;
                                 entry++) {
                                V[i * C * cols + buffer_offset + entry] =
                                    buffer[i * buffersize + entry];
                            }
                        }
                        buffer_entries = 0;
                    }
                }
            }
        }
    };

    const auto total = static_cast<size_t>(batch) * C;
    const auto nthreads = std::min<size_t>(cfg_num_threads, total);
    if (nthreads <= 1) {
        process_range(0, total);
    } else {
        Utils::ThreadGroup tg(m_evals_pool);
        for (auto t = size_t{0}; t < nthreads; t++) {
            const auto start = total * t / nthreads;
            const auto end = total * (t + 1) / nthreads;
            tg.add_task([&process_range, start, end]() {
                process_range(start, end);
            });
        }
        tg.wait_all();
    }
}

void CPUPipe::winograd_sgemm_batch(const std::vector<float>& U,
                                   const std::vector<float>& V,
                                   std::vector<float>& M, const int C,
                                   const int K, const int cols) {
    // cblas_sgemm_batch_strided would do all entries in one call, but the
    // batched API of this OpenBLAS build segfaults, so use the plain GEMM
    // per entry. OpenBLAS threads parallelize each call (measured ~187
    // GFLOPs/s on 6 threads for these sizes); the transform kernels below
    // are the single-threaded part and are parallelized in forward_batch.
#ifdef USE_BLAS
    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * cols;
        const auto offset_m = b * K * cols;
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, cols, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], cols,
                    0.0f,
                    &M[offset_m], cols);
    }
#else
    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * cols;
        const auto offset_m = b * K * cols;
        auto C_mat = EigenMatrixMap<float>(M.data() + offset_m, cols, K);
        C_mat.noalias() =
            ConstEigenMatrixMap<float>(V.data() + offset_v, cols, C)
            * ConstEigenMatrixMap<float>(U.data() + offset_u, K, C).transpose();
    }
#endif
}

void CPUPipe::winograd_transform_out_batch(const std::vector<float>& M,
                                           std::vector<float>& Y, const int K,
                                           const int batch) {
    constexpr auto W = BOARD_SIZE;
    constexpr auto H = BOARD_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;
    const auto cols = batch * P;

    const auto multiply_at = [](float& o0, float& o1, float& o2, float& o3,
                                const float i0, const float i1,
                                const float i2, const float i3,
                                const float i4, const float i5) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2 / 4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);
        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    // Each (b, k) output plane is independent, so parallelize over (b*K+k).
    const auto process_range = [&](const size_t start, const size_t end) {
        for (auto idx = start; idx < end; idx++) {
            const auto b = static_cast<int>(idx / K);
            const auto k = static_cast<int>(idx % K);
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto x = WINOGRAD_M * block_x;
                for (auto block_y = 0; block_y < WTILES; block_y++) {
                    const auto y = WINOGRAD_M * block_y;
                    const auto col = b * P + block_y * WTILES + block_x;

                    using WinogradTile =
                        std::array<std::array<float, WINOGRAD_ALPHA>,
                                   WINOGRAD_ALPHA>;
                    WinogradTile temp_m;
                    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                            temp_m[xi][nu] =
                                M[(xi * WINOGRAD_ALPHA + nu) * K * cols
                                  + k * cols + col];
                        }
                    }
                    std::array<std::array<float, WINOGRAD_ALPHA>,
                               WINOGRAD_ALPHA>
                        temp;
                    std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M> o;

                    for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                        multiply_at(temp[0][j], temp[1][j], temp[2][j],
                                    temp[3][j], temp_m[0][j], temp_m[1][j],
                                    temp_m[2][j], temp_m[3][j], temp_m[4][j],
                                    temp_m[5][j]);
                    }
                    for (auto i = 0; i < WINOGRAD_M; i++) {
                        multiply_at(o[i][0], o[i][1], o[i][2], o[i][3],
                                    temp[i][0], temp[i][1], temp[i][2],
                                    temp[i][3], temp[i][4], temp[i][5]);
                    }

                    const auto y_ind = (b * K + k) * H * W + y * W + x;
                    for (auto i = 0; i < WINOGRAD_M; i++) {
                        for (auto j = 0; j < WINOGRAD_M; j++) {
                            if (y + i < H && x + j < W) {
                                Y[y_ind + i * W + j] = o[i][j];
                            }
                        }
                    }
                }
            }
        }
    };

    const auto total = static_cast<size_t>(batch) * K;
    const auto nthreads = std::min<size_t>(cfg_num_threads, total);
    if (nthreads <= 1) {
        process_range(0, total);
    } else {
        Utils::ThreadGroup tg(m_evals_pool);
        for (auto t = size_t{0}; t < nthreads; t++) {
            const auto start = total * t / nthreads;
            const auto end = total * (t + 1) / nthreads;
            tg.add_task([&process_range, start, end]() {
                process_range(start, end);
            });
        }
        tg.wait_all();
    }
}

void CPUPipe::winograd_convolve3_batch(
    const int outputs, const std::vector<float>& input,
    const std::vector<float>& U, const int batch, const int cols,
    std::vector<float>& V, std::vector<float>& M,
    std::vector<float>& output) {
    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in_batch(input, V, input_channels, batch);
    winograd_sgemm_batch(U, V, M, input_channels, outputs, cols);
    winograd_transform_out_batch(M, output, outputs, batch);
}

template <unsigned int filter_size>
void convolve(const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time
    constexpr unsigned int width = BOARD_SIZE;
    constexpr unsigned int height = BOARD_SIZE;
    constexpr auto num_intersections = width * height;
    constexpr auto filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * num_intersections == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);
    // Always Eigen (see winograd_sgemm): the 1x1 head convs are small and run
    // in parallel from the search threads.
    auto C_mat =
        EigenMatrixMap<float>(output.data(), num_intersections, outputs);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(col.data(), num_intersections, filter_dim)
        * ConstEigenMatrixMap<float>(weights.data(), filter_dim, outputs);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < num_intersections; b++) {
            output[(o * num_intersections) + b] += biases[o];
        }
    }
}

void CPUPipe::forward(const std::vector<float>& input,
                      std::vector<float>& output_pol,
                      std::vector<float>& output_val) {
    // Input convolution
    constexpr auto P = WINOGRAD_P;
    // Calculate output channels
    const auto output_channels = m_input_channels;
    // input_channels is the maximum number of input channels of any
    // convolution. Residual blocks are identical, but the first convolution
    // might be bigger when the network has very few filters
    const auto input_channels =
        std::max(static_cast<size_t>(output_channels),
                 static_cast<size_t>(Network::INPUT_CHANNELS));
    auto conv_out = std::vector<float>(output_channels * NUM_INTERSECTIONS);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * P);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * P);

    // Input conv. Two architectures (distinguished by m_input_has_bn):
    //  - tfprocess/GoNet training graphs: conv -> bias -> BN[0] -> ReLU.
    //  - Real PhoenixGo graph: conv -> bias only (BN[0] in the weight file is
    //    identity and must NOT be applied — applying it would insert a spurious
    //    BN+ReLU and corrupt the network output).
    winograd_convolve3(output_channels, input, m_weights->m_conv_weights[0], V,
                       M, conv_out);
    if (m_weights->m_input_has_bn
        && !m_weights->m_batchnorm_means.empty()
        && m_weights->m_batchnorm_means[0].size() == output_channels) {
        batchnorm<NUM_INTERSECTIONS>(output_channels, conv_out,
                                     m_weights->m_batchnorm_means[0].data(),
                                     m_weights->m_batchnorm_stddevs[0].data());
    }

    // Residual tower (pre-activation: BN+ReLU BEFORE conv)
    // PG block: BN+ReLU(x) -> conv -> BN+ReLU(c0) -> conv -> +residual
    //           (NO ReLU after residual add)
    auto conv_in = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    auto res = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    for (auto i = size_t{1}; i < m_weights->m_conv_weights.size(); i += 2) {
        const auto channels = m_input_channels;
        // Save shortcut = current input to res block (before any BN)
        std::copy(conv_out.begin(), conv_out.end(), res.begin());

        // cb0: BN+ReLU(x) -> conv
        batchnorm<NUM_INTERSECTIONS>(channels, conv_out,
                                     m_weights->m_batchnorm_means[i].data(),
                                     m_weights->m_batchnorm_stddevs[i].data());
        std::swap(conv_out, conv_in);
        winograd_convolve3(channels, conv_in,
                           m_weights->m_conv_weights[i], V, M, conv_out);

        // cb1: BN+ReLU(c0) -> conv -> +residual (no ReLU after add)
        batchnorm<NUM_INTERSECTIONS>(channels, conv_out,
                                     m_weights->m_batchnorm_means[i + 1].data(),
                                     m_weights->m_batchnorm_stddevs[i + 1].data());
        std::swap(conv_out, conv_in);
        winograd_convolve3(channels, conv_in,
                           m_weights->m_conv_weights[i + 1], V, M, conv_out);

        // Add residual without ReLU (PhoenixGo: no ReLU after residual add)
        for (auto b = size_t{0}; b < conv_out.size(); b++) {
            conv_out[b] += res[b];
        }
    }
    // PhoenixGo: trunk BN + ReLU after residual tower, before heads.
    // PG network has layer_final/batch_norm; LZ original does not.
    // Empty for v1/v2 networks (no-op). batchnorm<>() includes ReLU (max(0,...)),
    // matching PG's trunk BN -> ReLU -> head_conv flow.
    if (!m_weights->m_bn_trunk_means.empty()) {
        batchnorm<NUM_INTERSECTIONS>(output_channels, conv_out,
                                     m_weights->m_bn_trunk_means.data(),
                                     m_weights->m_bn_trunk_stddevs.data());
    }
    convolve<1>(Network::OUTPUTS_POLICY, conv_out, m_conv_pol_w, m_conv_pol_b,
                output_pol);
    convolve<1>(Network::OUTPUTS_VALUE, conv_out, m_conv_val_w, m_conv_val_b,
                output_val);
}

void CPUPipe::forward_batch(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& output_pol,
    std::vector<std::vector<float>>& output_val) {
    // Batched (KataGo-style) forward: all positions run through the winograd
    // SGEMMs together, so the (large) weight set is read once per batch
    // instead of once per position. Combined with multi-threaded BLAS this is
    // what makes CPU inference competitive with GPU for big networks.
    const auto batch = static_cast<int>(inputs.size());
    const auto C = m_input_channels;
    const auto cols = batch * WINOGRAD_P;

    // Buffer capacity: the input conv takes INPUT_CHANNELS input planes, all
    // residual convs take `channels`.
    const auto maxC = static_cast<int>(std::max(
        static_cast<size_t>(C), static_cast<size_t>(Network::INPUT_CHANNELS)));
    auto V = std::vector<float>(static_cast<size_t>(WINOGRAD_TILE) * maxC
                                * cols);
    auto M = std::vector<float>(static_cast<size_t>(WINOGRAD_TILE) * C
                                * cols);

    auto in = std::vector<float>(static_cast<size_t>(batch)
                                 * Network::INPUT_CHANNELS
                                 * NUM_INTERSECTIONS);
    for (auto b = 0; b < batch; b++) {
        std::copy(inputs[b]->begin(), inputs[b]->end(),
                  in.begin() + b * Network::INPUT_CHANNELS
                      * NUM_INTERSECTIONS);
    }
    auto conv_out = std::vector<float>(static_cast<size_t>(batch) * C
                                       * NUM_INTERSECTIONS);
    auto conv_in = std::vector<float>(static_cast<size_t>(batch) * C
                                      * NUM_INTERSECTIONS);
    auto res = std::vector<float>(static_cast<size_t>(batch) * C
                                  * NUM_INTERSECTIONS);

    // BN + ReLU in place over [batch][C][361], parallelized over (b,c).
    const auto batchnorm_batch = [&](std::vector<float>& data,
                                     const float* const means,
                                     const float* const stddevs) {
        const auto total = static_cast<size_t>(batch) * C;
        const auto nthreads = std::min<size_t>(cfg_num_threads, total);
        const auto run = [&](const size_t start, const size_t end) {
            for (auto idx = start; idx < end; idx++) {
                const auto b = static_cast<int>(idx / C);
                const auto c = static_cast<int>(idx % C);
                const auto mean = means[c];
                const auto scale = stddevs[c];
                auto arr = &data[(static_cast<size_t>(b) * C + c)
                                 * NUM_INTERSECTIONS];
                for (auto pos = 0; pos < NUM_INTERSECTIONS; pos++) {
                    arr[pos] = std::max(0.0f, scale * (arr[pos] - mean));
                }
            }
        };
        if (nthreads <= 1) {
            run(0, total);
        } else {
            Utils::ThreadGroup tg(m_evals_pool);
            for (auto t = size_t{0}; t < nthreads; t++) {
                const auto start = total * t / nthreads;
                const auto end = total * (t + 1) / nthreads;
                tg.add_task([&run, start, end]() { run(start, end); });
            }
            tg.wait_all();
        }
    };

    // Input conv: apply BN[0] only when the graph actually has one (see the
    // comment in forward(); the real PhoenixGo graph has no input BN).
    winograd_convolve3_batch(C, in, m_weights->m_conv_weights[0], batch, cols,
                             V, M, conv_out);
    if (m_weights->m_input_has_bn
        && !m_weights->m_batchnorm_means.empty()
        && m_weights->m_batchnorm_means[0].size()
               == static_cast<size_t>(C)) {
        batchnorm_batch(conv_out, m_weights->m_batchnorm_means[0].data(),
                        m_weights->m_batchnorm_stddevs[0].data());
    }

    // Residual tower (pre-activation): BN+ReLU -> conv -> BN+ReLU -> conv ->
    // +residual (no ReLU after the add).
    for (auto i = size_t{1}; i < m_weights->m_conv_weights.size(); i += 2) {
        std::copy(conv_out.begin(), conv_out.end(), res.begin());
        batchnorm_batch(conv_out, m_weights->m_batchnorm_means[i].data(),
                        m_weights->m_batchnorm_stddevs[i].data());
        std::swap(conv_out, conv_in);
        winograd_convolve3_batch(C, conv_in, m_weights->m_conv_weights[i],
                                 batch, cols, V, M, conv_out);
        batchnorm_batch(conv_out, m_weights->m_batchnorm_means[i + 1].data(),
                        m_weights->m_batchnorm_stddevs[i + 1].data());
        std::swap(conv_out, conv_in);
        winograd_convolve3_batch(C, conv_in,
                                 m_weights->m_conv_weights[i + 1], batch,
                                 cols, V, M, conv_out);
        // Residual add (flat, parallelized).
        {
            const auto total = conv_out.size();
            const auto nthreads =
                std::min<size_t>(cfg_num_threads, total);
            if (nthreads <= 1) {
                for (auto b2 = size_t{0}; b2 < total; b2++) {
                    conv_out[b2] += res[b2];
                }
            } else {
                Utils::ThreadGroup tg(m_evals_pool);
                for (auto t = size_t{0}; t < nthreads; t++) {
                    const auto start = total * t / nthreads;
                    const auto end = total * (t + 1) / nthreads;
                    tg.add_task([&, start, end]() {
                        for (auto b2 = start; b2 < end; b2++) {
                            conv_out[b2] += res[b2];
                        }
                    });
                }
                tg.wait_all();
            }
        }
    }

    // Trunk BN + ReLU (PhoenixGo-specific, before the heads).
    if (!m_weights->m_bn_trunk_means.empty()) {
        batchnorm_batch(conv_out, m_weights->m_bn_trunk_means.data(),
                        m_weights->m_bn_trunk_stddevs.data());
    }

    // 1x1 head convs over the batch (cheap): out[b][o][pos] =
    // sum_c W[o*C+c] * in[(b*C+c)*361+pos]. Head BN/dense stay on the host.
    const auto convolve1_batch = [&](const std::vector<float>& data,
                                     const std::vector<float>& W,
                                     const int outputs,
                                     std::vector<float>& out) {
        out.resize(static_cast<size_t>(batch) * outputs
                   * NUM_INTERSECTIONS);
        const auto total = static_cast<size_t>(batch) * outputs;
        const auto nthreads = std::min<size_t>(cfg_num_threads, total);
        const auto run = [&](const size_t start, const size_t end) {
            for (auto idx = start; idx < end; idx++) {
                const auto b = static_cast<int>(idx / outputs);
                const auto o = static_cast<int>(idx % outputs);
                for (auto pos = 0; pos < NUM_INTERSECTIONS; pos++) {
                    auto acc = 0.0f;
                    for (auto c = 0; c < C; c++) {
                        acc += W[o * C + c]
                               * data[(static_cast<size_t>(b) * C + c)
                                          * NUM_INTERSECTIONS + pos];
                    }
                    out[(static_cast<size_t>(b) * outputs + o)
                            * NUM_INTERSECTIONS + pos] = acc;
                }
            }
        };
        if (nthreads <= 1) {
            run(0, total);
        } else {
            Utils::ThreadGroup tg(m_evals_pool);
            for (auto t = size_t{0}; t < nthreads; t++) {
                const auto start = total * t / nthreads;
                const auto end = total * (t + 1) / nthreads;
                tg.add_task([&run, start, end]() { run(start, end); });
            }
            tg.wait_all();
        }
    };
    std::vector<float> pol;
    std::vector<float> val;
    convolve1_batch(conv_out, m_conv_pol_w, Network::OUTPUTS_POLICY, pol);
    convolve1_batch(conv_out, m_conv_val_w, Network::OUTPUTS_VALUE, val);

    output_pol.resize(batch);
    output_val.resize(batch);
    for (auto b = 0; b < batch; b++) {
        output_pol[b].assign(
            pol.begin() + b * Network::OUTPUTS_POLICY * NUM_INTERSECTIONS,
            pol.begin() + (b + 1) * Network::OUTPUTS_POLICY
                * NUM_INTERSECTIONS);
        output_val[b].assign(
            val.begin() + b * Network::OUTPUTS_VALUE * NUM_INTERSECTIONS,
            val.begin() + (b + 1) * Network::OUTPUTS_VALUE
                * NUM_INTERSECTIONS);
    }
}

void CPUPipe::push_weights(const unsigned int /*filter_size*/,
                           const unsigned int /*channels*/,
                           const unsigned int outputs,
                           std::shared_ptr<const ForwardPipeWeights> weights) {

    m_weights = weights;

    // Output head convolutions
    m_conv_pol_w = weights->m_conv_pol_w;
    m_conv_pol_b.resize(m_conv_pol_w.size() / outputs, 0.0f);
    m_conv_val_w = weights->m_conv_val_w;
    m_conv_val_b.resize(m_conv_val_w.size() / outputs, 0.0f);
}
