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

#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED

#include "config.h"

#include <chrono>
#include <cstddef>
#include <cstdint>

using std::chrono::operator""ms;
using std::chrono::operator""s;
using TimePoint = std::chrono::steady_clock::time_point;

struct Timer {
    std::uint64_t start;
    Timer();
    std::uint64_t elapsed() const;
    std::uint64_t elapsed_ms() const;
};

std::uint64_t cpuid_cycles();
std::uint64_t Timer_tsc_to_ms(const std::uint64_t duration);
std::uint64_t cfg_to_stones(int playouts);

#endif
