#ifndef FULLBOARD_H_INCLUDED
#define FULLBOARD_H_INCLUDED

#include "FastBoard.h"

#include <string>
#include <cstdint>

class FullBoard : public FastBoard {
public:
    FullBoard() = default;

    int estimate_score(float komi) const;

    std::string move_to_text_sgf(const int move) const;
    void display_board(const int lastmove);

    std::uint64_t calc_hash(const int komove, bool superko = false) const;
    std::uint64_t calc_symmetry_hash(const int komove, int symmetry) const;
};

#endif