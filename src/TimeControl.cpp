/* TimeControl.cpp - Game time management */
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <regex>
#include <sstream>
#include "TimeControl.h"
#include "GTP.h"
#include "Timing.h"
#include "Utils.h"
using namespace Utils;
TimeControl::TimeControl(const int maintime, const int byotime,
                         const int byostones, const int byoperiods)
    : m_maintime(maintime), m_byotime(byotime), m_byostones(byostones),
      m_byoperiods(byoperiods) {
    reset_clocks();
}
std::string TimeControl::stones_left_to_text_sgf(const int color) const {
    auto s = std::string{};
    if (m_inbyo[color]) {
        const auto c = color == FastBoard::BLACK ? "OB[" : "OW[";
        if (m_byostones) {
            s += c + std::to_string(m_stones_left[color]) + "]";
        } else if (m_byoperiods) {
            s += c + std::to_string(m_periods_left[color]) + "]";
        }
    }
    return s;
}
std::string TimeControl::to_text_sgf() const {
    if (m_byotime != 0 && m_byostones == 0 && m_byoperiods == 0) {
        return "";
    }
    auto s = "TM[" + std::to_string(m_maintime / 100) + "]";
    if (m_byotime) {
        if (m_byostones) {
            s += "OT[" + std::to_string(m_byostones) + "/";
            s += std::to_string(m_byotime / 100) + " Canadian]";
        } else {
            assert(m_byoperiods);
            s += "OT[" + std::to_string(m_byoperiods) + "x";
            s += std::to_string(m_byotime / 100) + " byo-yomi]";
        }
        s += stones_left_to_text_sgf(FastBoard::BLACK);
        s += stones_left_to_text_sgf(FastBoard::WHITE);
    }
    const auto black_time_left =
        (m_remaining_time[FastBoard::BLACK] + 99) / 100;
    const auto white_time_left =
        (m_remaining_time[FastBoard::WHITE] + 99) / 100;
    s += "BL[" + std::to_string(black_time_left) + "]";
    s += "WL[" + std::to_string(white_time_left) + "]";
    return s;
}
std::shared_ptr<TimeControl> TimeControl::make_from_text_sgf(
    const std::string& maintime, const std::string& byoyomi,
    const std::string& black_time_left, const std::string& white_time_left,
    const std::string& black_moves_left, const std::string& white_moves_left) {
    const auto maintime_centis = std::stoi(maintime) * 100;
    auto byotime = 0;
    auto byostones = 0;
    auto byoperiods = 0;
    if (!byoyomi.empty()) {
        std::smatch m;
        const auto re_canadian = std::regex{"(\\d+)/(\\d+) Canadian"};
        const auto re_byoyomi = std::regex{"(\\d+)x(\\d+) byo-yomi"};
        if (std::regex_match(byoyomi, m, re_canadian)) {
            byostones = std::stoi(m[1]);
            byotime = std::stoi(m[2]) * 100;
        } else if (std::regex_match(byoyomi, m, re_byoyomi)) {
            byoperiods = std::stoi(m[1]);
            byotime = std::stoi(m[2]) * 100;
        }
    }
    const auto timecontrol_ptr = std::make_shared<TimeControl>(
        maintime_centis, byotime, byostones, byoperiods);
    if (!black_time_left.empty()) {
        const auto time = std::stoi(black_time_left) * 100;
        const auto stones =
            black_moves_left.empty() ? 0 : std::stoi(black_moves_left);
        timecontrol_ptr->adjust_time(FastBoard::BLACK, time, stones);
    }
    if (!white_time_left.empty()) {
        const auto time = std::stoi(white_time_left) * 100;
        const auto stones =
            white_moves_left.empty() ? 0 : std::stoi(white_moves_left);
        timecontrol_ptr->adjust_time(FastBoard::WHITE, time, stones);
    }
    return timecontrol_ptr;
}
void TimeControl::reset_clocks() {
    m_remaining_time = {m_maintime, m_maintime};
    m_stones_left = {m_byostones, m_byostones};
    m_periods_left = {m_byoperiods, m_byoperiods};
    m_inbyo = {m_maintime <= 0, m_maintime <= 0};
    if (m_inbyo[0]) {
        m_remaining_time[0] = m_byotime;
    }
    if (m_inbyo[1]) {
        m_remaining_time[1] = m_byotime;
    }
}
void TimeControl::start(const int color) {
    m_times[color] = Time();
}
void TimeControl::stop(const int color) {
    Time stop;
    int elapsed_centis = Time::timediff_centis(m_times[color], stop);
    assert(elapsed_centis >= 0);
    m_remaining_time[color] -= elapsed_centis;
    if (m_inbyo[color]) {
        if (m_byostones) {
            m_stones_left[color]--;
        } else if (m_byoperiods) {
            if (elapsed_centis > m_byotime) {
                m_periods_left[color]--;
            }
        }
    }
    if (!m_inbyo[color] && m_remaining_time[color] <= 0) {
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
        m_periods_left[color] = m_byoperiods;
        m_inbyo[color] = true;
    } else if (m_inbyo[color] && m_byostones && m_stones_left[color] <= 0) {
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
    } else if (m_inbyo[color] && m_byoperiods) {
        m_remaining_time[color] = m_byotime;
    }
}
void TimeControl::display_color_time(const int color) {
    auto rem = m_remaining_time[color] / 100;
    auto minuteDiv = std::div(rem, 60);
    auto hourDiv = std::div(minuteDiv.quot, 60);
    auto seconds = minuteDiv.rem;
    auto minutes = hourDiv.rem;
    auto hours = hourDiv.quot;
    auto name = color == 0 ? "Black" : "White";
    myprintf("%s time: %02d:%02d:%02d", name, hours, minutes, seconds);
    if (m_inbyo[color]) {
        if (m_byostones) {
            myprintf(", %d stones left", m_stones_left[color]);
        } else if (m_byoperiods) {
            myprintf(", %d period(s) of %d seconds left",
                     m_periods_left[color], m_byotime / 100);
        }
    }
    myprintf("\n");
}
void TimeControl::display_times() {
    display_color_time(FastBoard::BLACK);
    display_color_time(FastBoard::WHITE);
    myprintf("\n");
}
int TimeControl::max_time_for_move(const int boardsize, const int color,
                                   const size_t movenum) const {
    auto time_remaining = m_remaining_time[color];
    auto moves_remaining = get_moves_expected(boardsize, movenum);
    auto extra_time_per_move = 0;
    if (m_byotime != 0) {
        if (m_byostones == 0 && m_byoperiods == 0) {
            return 31 * 24 * 60 * 60 * 100;
        }
        if (m_inbyo[color]) {
            if (m_byostones) {
                moves_remaining = m_stones_left[color];
            } else {
                assert(m_byoperiods);
                time_remaining = 0;
                extra_time_per_move = m_byotime;
            }
        } else {
            if (m_byostones) {
                int byo_extra = m_byotime / m_byostones;
                time_remaining = m_remaining_time[color] + byo_extra;
                extra_time_per_move = byo_extra;
            } else {
                assert(m_byoperiods);
                int byo_extra = m_byotime * (m_periods_left[color] - 1);
                time_remaining = m_remaining_time[color] + byo_extra;
                extra_time_per_move = m_byotime;
            }
        }
    }
    auto base_time = std::max(time_remaining - cfg_lagbuffer_cs, 0)
                     / std::max(moves_remaining, 1);
    auto inc_time = std::max(extra_time_per_move - cfg_lagbuffer_cs, 0);
    return base_time + inc_time;
}
void TimeControl::adjust_time(const int color, const int time,
                              const int stones) {
    m_remaining_time[color] = time;
    if (!time && !stones) {
        m_inbyo[color] = true;
        m_remaining_time[color] = m_byotime;
        m_stones_left[color] = m_byostones;
        m_periods_left[color] = m_byoperiods;
    }
    if (stones) {
        m_inbyo[color] = true;
    }
    if (m_inbyo[color]) {
        if (m_byostones) {
            m_stones_left[color] = stones;
        } else if (m_byoperiods) {
            m_periods_left[color] = stones;
        }
    }
}
size_t TimeControl::opening_moves(const int boardsize) const {
    auto num_intersections = boardsize * boardsize;
    auto fast_moves = num_intersections / 6;
    return fast_moves;
}
int TimeControl::get_moves_expected(const int boardsize,
                                    const size_t movenum) const {
    auto board_div = 5;
    if (cfg_timemanage != TimeManagement::OFF) {
        board_div = 9;
    }
    auto base_remaining = (boardsize * boardsize) / board_div;
    auto fast_moves = opening_moves(boardsize);
    if (movenum < fast_moves) {
        return (base_remaining + fast_moves) - movenum;
    } else {
        return base_remaining;
    }
}
bool TimeControl::can_accumulate_time(const int color) const {
    if (m_inbyo[color]) {
        if (m_byoperiods) {
            return false;
        }
        if (m_byostones && m_stones_left[color] == 1) {
            return false;
        }
    }
    return true;
}
