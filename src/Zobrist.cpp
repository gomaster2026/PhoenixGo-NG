#include "config.h"
#include "Zobrist.h"
#include "Random.h"
std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES>,     4> Zobrist::zobrist;
std::array<std::uint64_t, FastBoard::NUM_VERTICES>                    Zobrist::zobrist_ko;
std::array<std::array<std::uint64_t, FastBoard::NUM_VERTICES * 2>, 2> Zobrist::zobrist_pris;
std::array<std::uint64_t, 5>                                          Zobrist::zobrist_pass;
void Zobrist::init_zobrist(Random& rng) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < FastBoard::NUM_VERTICES; j++) {
            Zobrist::zobrist[i][j] = rng.randuint64();
        }
    }
    for (int j = 0; j < FastBoard::NUM_VERTICES; j++) {
        Zobrist::zobrist_ko[j] = rng.randuint64();
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < FastBoard::NUM_VERTICES * 2; j++) {
            Zobrist::zobrist_pris[i][j] = rng.randuint64();
        }
    }
    for (int i = 0; i < 5; i++) {
        Zobrist::zobrist_pass[i] = rng.randuint64();
    }
}
