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

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "FullBoard.h"

#include "Utils.h"

using namespace Utils;

int FullBoard::estimate_score(float komi) const {
    auto score = 0;
    auto bd = std::vector<bool>(m_numvertices, false);
    auto open = std::queue<int>();

    for (auto i = 0; i < m_boardsize; i++) {
        for (auto j = 0; j < m_boardsize; j++) {
            auto vertex = get_vertex(i, j);
            if (m_state[vertex] == WHITE) {
                score--;
                bd[vertex] = true;
            } else if (m_state[vertex] == BLACK) {
                score++;
                bd[vertex] = true;
            }
        }
    }

    // flood fill regions to count territory
    for (auto i = 0; i < m_boardsize; i++) {
        for (auto j = 0; j < m_boardsize; j++) {
            auto vertex = get_vertex(i, j);
            if (!bd[vertex] && m_state[vertex] == EMPTY) {
                // start flood fill
                auto st = 0;
                auto blacksc = false;
                auto whitesc = false;
                open.push(vertex);
                bd[vertex] = true;
                auto region = std::vector<int>();
                region.push_back(vertex);

                while (!open.empty()) {
                    auto v = open.front();
                    open.pop();
                    st++;

                    for (auto k = 0; k < 4; k++) {
                        auto nv = v + m_dirs[k];
                        if (!bd[nv]) {
                            if (m_state[nv] == WHITE) {
                                whitesc = true;
                            } else if (m_state[nv] == BLACK) {
                                blacksc = true;
                            } else {
                                bd[nv] = true;
                                open.push(nv);
                                region.push_back(nv);
                            }
                        }
                    }
                }

                if (!blacksc && !whitesc) {
                    // seki
                } else if (blacksc && whitesc) {
                    // dame
                } else if (blacksc) {
                    score += st;
                } else {
                    score -= st;
                }
            }
        }
    }

    // apply komi
    score -= static_cast<int>(komi + 0.5f);

    return score;
}

std::string FullBoard::move_to_text_sgf(const int move) const {
    return FastBoard::move_to_text_sgf(move);
}

void FullBoard::display_board(const int lastmove) {
    int boardsize = get_boardsize();

    myprintf("\n   ");
    print_columns();
    for (int j = boardsize - 1; j >= 0; j--) {
        myprintf("%2d", j + 1);
        if (lastmove == get_vertex(0, j))
            myprintf("(");
        else
            myprintf(" ");
        for (int i = 0; i < boardsize; i++) {
            if (get_state(i, j) == WHITE) {
                myprintf("O");
            } else if (get_state(i, j) == BLACK) {
                myprintf("X");
            } else if (starpoint(boardsize, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            if (lastmove == get_vertex(i, j)) {
                myprintf(")");
            } else if (i != boardsize - 1 && lastmove == get_vertex(i, j) + 1) {
                myprintf("(");
            } else {
                myprintf(" ");
            }
        }
        myprintf("%2d\n", j + 1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
}

void FullBoard::print_columns() {
    for (int i = 0; i < get_boardsize(); i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25)
                                                    : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}