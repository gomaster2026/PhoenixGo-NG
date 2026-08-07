#ifndef FULLBOARD_H_INCLUDED
#define FULLBOARD_H_INCLUDED
#include "config.h"
#include <cstdint>
#include "FastBoard.h"
class FullBoard : public FastBoard {
public:
    int remove_string(int i);
    int update_board(int color, int i);
    std::uint64_t get_hash() const;
    std::uint64_t get_ko_hash() const;
    void set_to_move(int tomove);
    void reset_board(int size);
    void display_board(int lastmove = -1);
    std::uint64_t calc_hash(int komove = NO_VERTEX) const;
    std::uint64_t calc_symmetry_hash(int komove, int symmetry) const;
    std::uint64_t calc_ko_hash() const;
    std::uint64_t m_hash;
    std::uint64_t m_ko_hash;
private:
    template <class Function>
    std::uint64_t calc_hash(int komove, Function transform) const;
};
#endif
