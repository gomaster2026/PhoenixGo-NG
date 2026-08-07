#ifndef TUNER_H_INCLUDED
#define TUNER_H_INCLUDED

#include "config.h"
#include <vector>

#ifdef USE_OPENCL
class Tuner {
public:
    static void run_tuning_round(const std::string& name,
                                 std::vector<unsigned int> m_global_size,
                                 std::vector<unsigned int> m_local_size,
                                 const std::vector<std::string>& m_src,
                                 std::vector<long> m_tuneparams);
    Tuner m_tuner;
};
#endif

#endif
