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

// OpenCLScheduler 是模板类，实现全在 OpenCLScheduler.h 中。
// 此 .cpp 文件仅为 CMake glob 收集而保留（非 USE_CPU_ONLY 编译时
// 会被包含进编译列表，但无实际代码需要编译）。

#include "config.h"
#include "OpenCLScheduler.h"
