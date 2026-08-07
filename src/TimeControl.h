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

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include "config.h"

#include <cstdint>
#include <vector>

class TimeControl {
public:
    void reset_clocks();
    TimeControl(int fischer_time, int fischer_inc,
                int fischer_per_move_ponder, int fischer_per_move_noponder,
                bool infinite_time);
    TimeControl(int main_time, int byo_yomi_time, int byo_yomi_stones,
                int byo_yomi_periods, bool main_time_enabled,
                bool infinite_time);
    TimeControl() = default;
    void start(int color);
    bool-ended(int color);
    float remaining(int color);
    float elapsed(void);
    void set_clocks(int main_time, int byo_yomi_time, int byo_yomi_stones,
                    int byo_yomi_periods);
    void set_fischer(int fischer_time, int fischer_inc,
                     int fischer_per_move_ponder,
                     int fischer_per_move_noponder);
    int time_left(int color);
    int get_stones_remaining(int color);
    int get_time_per_move(int color);
    bool is_infinite() const { return m_infinite_time; }
    int get_main_time(int color) const { return m_time_left[color]; }
    int get_byo_yomi_time(int color) const { return m_byo_yomi_time_left[color]; }
    int get_byo_yomi_stones(int color) const { return m_stones_left[color]; }
    int get_byo_yomi_periods(int color) const { return m_periods_left[color]; }
private:
    bool m_is_fischer{false};
    int m_max_time_for_move{0};
    int m_fischer_time{0};
    int m_fischer_inc{0};
    int m_fischer_per_move_ponder{0};
    int m_fischer_per_move_noponder{0};
    bool m_infinite_time{false};
    std::vector<int> m_time_left;
    std::vector<int> m_stones_left;
    std::vector<int> m_byo_yomi_time_left;
    std::vector<int> m_periods_left;
    int m_byo_yomi_time{0};
    int m_byo_yomi_stones{0};
    int m_byo_yomi_periods{0};
    bool m_main_time_enabled{false};
    std::uint64_t m_last_clock_time{0};
};

#endif
