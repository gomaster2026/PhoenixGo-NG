#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED
#include "config.h"
#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>
class FastBoard {
    friend class FastState;
public:
    static constexpr int NBR_SHIFT = 4;
    static constexpr int NBR_MASK = (1 << NBR_SHIFT) - 1;
    static constexpr int NUM_VERTICES = ((BOARD_SIZE + 2) * (BOARD_SIZE + 2));
    static constexpr int NO_VERTEX = 0;
    static constexpr int PASS = -1;
    static constexpr int RESIGN = -2;
    enum vertex_t : char { BLACK = 0, WHITE = 1, EMPTY = 2, INVAL = 3 };
    int get_boardsize() const;
    vertex_t get_state(int x, int y) const;
    vertex_t get_state(int vertex) const;
    int get_vertex(int x, int y) const;
    void set_state(int x, int y, vertex_t content);
    void set_state(int vertex, vertex_t content);
    std::pair<int, int> get_xy(int vertex) const;
    bool is_suicide(int i, int color) const;
    int count_pliberties(int i) const;
    bool is_eye(int color, int vtx) const;
    float area_score(float komi) const;
    int get_prisoners(int side) const;
    bool black_to_move() const;
    bool white_to_move() const;
    int get_to_move() const;
    void set_to_move(int color);
    std::string move_to_text(int move) const;
    int text_to_move(std::string move) const;
    std::string move_to_text_sgf(int move) const;
    std::string get_stone_list() const;
    std::string get_string(int vertex) const;
    void reset_board(int size);
    void display_board(int lastmove = -1);
    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);
protected:
    static const std::array<int,      2> s_eyemask;
    static const std::array<vertex_t, 4> s_cinvert;
    std::array<vertex_t, NUM_VERTICES>           m_state;
    std::array<unsigned short, NUM_VERTICES + 1> m_next;
    std::array<unsigned short, NUM_VERTICES + 1> m_parent;
    std::array<unsigned short, NUM_VERTICES + 1> m_libs;
    std::array<unsigned short, NUM_VERTICES + 1> m_stones;
    std::array<unsigned short, NUM_VERTICES>     m_neighbours;
    std::array<int, 4>                           m_dirs;
    std::array<int, 2>                           m_prisoners;
    std::array<unsigned short, NUM_VERTICES>     m_empty;
    std::array<unsigned short, NUM_VERTICES>     m_empty_idx;
    int m_empty_cnt;
    int m_tomove;
    int m_numvertices;
    int m_boardsize;
    int m_sidevertices;
    int calc_reach_color(int color) const;
    int count_neighbours(int color, int i) const;
    void merge_strings(int ip, int aip);
    void add_neighbour(int i, int color);
    void remove_neighbour(int i, int color);
    void print_columns();
};
#endif
