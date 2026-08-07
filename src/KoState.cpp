#include "config.h"

#include <algorithm>
#include <cassert>
#include <iterator>

#include "KoState.h"

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"

void KoState::init_game(const int size, const float komi) {
    assert(size <= BOARD_SIZE);
    FastState::init_game(size, komi);
    m_ko_hash_history.clear();
    m_ko_hash_history.emplace_back(board.get_ko_hash());
}

bool KoState::superko() const {
    auto first = crbegin(m_ko_hash_history);
    auto last = crend(m_ko_hash_history);
    auto res = std::find(++first, last, board.get_ko_hash());
    return (res != last);
}

void KoState::reset_game() {
    FastState::reset_game();
    m_ko_hash_history.clear();
    m_ko_hash_history.push_back(board.get_ko_hash());
}

void KoState::play_move(const int vertex) {
    play_move(board.get_to_move(), vertex);
}

void KoState::play_move(const int color, const int vertex) {
    if (vertex != FastBoard::RESIGN) {
        FastState::play_move(color, vertex);
    }
    m_ko_hash_history.push_back(board.get_ko_hash());
}
