/* Timing.cpp - Time measurement utilities */
#include <chrono>
#include "Timing.h"
int Time::timediff_centis(const Time start, const Time end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (end.m_time - start.m_time).count() / 10;
}
double Time::timediff_seconds(const Time start, const Time end) {
    return std::chrono::duration<double>(end.m_time - start.m_time).count();
}
Time::Time() {
    m_time = std::chrono::steady_clock::now();
}
