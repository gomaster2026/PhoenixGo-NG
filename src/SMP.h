/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

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

#ifndef SMP_H_INCLUDED
#define SMP_H_INCLUDED

#include "config.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <mutex>

class Spinlock {
public:
    void lock() {
        while (m_lock.exchange(true, std::memory_order_acquire)) {
#if defined(_MSC_VER)
            _mm_pause();
#elif defined(__i386__)
            __asm__ __volatile__("pause");
#elif defined(__aarch64__)
            __asm__ __volatile__("hint #0x0");
#endif
        }
    }
    bool try_lock() {
        return !m_lock.exchange(true, std::memory_order_acquire);
    }
    void unlock() {
        m_lock.store(false, std::memory_order_release);
    }
private:
    std::atomic<bool> m_lock{false};
};

#endif
