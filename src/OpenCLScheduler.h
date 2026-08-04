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

#ifndef OPENCLSCHEDULER_H_INCLUDED
#define OPENCLSCHEDULER_H_INCLUDED

#include "config.h"
#include "CPUPipe.h"

#ifdef USE_HALF
#include "half/half.hpp"
#endif

// OpenCLScheduler: 模板类，模板参数为 float 或 half_float::half。
//
// 此实现内部委托给 CPUPipe（CPU 计算），提供 OpenCL 接口兼容性，
// 使非 USE_CPU_ONLY 编译能通过。适用于无 OpenCL 驱动或驱动不可用的环境。
//
// Network.cpp 中的使用方式:
//   OpenCLScheduler<float>()          // 单精度
//   OpenCLScheduler<half_float::half>()  // 半精度
//
// needs_autodetect() 返回 false，跳过 benchmark，直接使用指定精度。
// 由于实际计算由 CPUPipe 完成（float），模板参数不影响计算结果。
template<typename T>
class OpenCLScheduler : public CPUPipe {
public:
    // 返回 false 表示不需要自动检测精度。
    // Network.cpp select_precision() 会根据此值决定是否 benchmark。
    // 返回 false 跳过 benchmark，直接使用当前精度。
    bool needs_autodetect() override {
        return false;
    }
};

#endif
