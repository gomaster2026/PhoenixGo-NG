#ifndef TRAINING_H_INCLUDED
#define TRAINING_H_INCLUDED

#include "config.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "GameState.h"
#include "Network.h"

class Training {
public:
    void dump_training(const std::string& filename, bool compress);
    void dump_scored_moves(const std::string& filename, bool compress);
    void reset(void);
    void record(const GameState& state, const Network::Netresult& result);
private:
    struct MoveRecord {
        std::array<float, NUM_INTERSECTIONS> policy;
        float winrate_pass;
        float winrate;
        float eval;
        int to_move;
        std::vector<int> board_features;
        std::vector<int> move_features;
        bool is_black_to_move;
    };
    std::vector<MoveRecord> m_data;
    struct OutputChunk {
        std::int32_t num_samples;
        std::int32_t label;
        std::int32_t is_black_to_move;
        std::int32_t rule;
        std::int32_t ko;
        std::float32_t eval;
        std::float32_t deadstones;
        float policies[NUM_INTERSECTIONS * POTENTIAL_MOVES];
    } m_output;
    struct OutputChunk2 {
        std::int32_t num_samples;
        std::int32_t rule;
        std::int32_t ko;
        std::int32_t pad;
        std::float32_t eval;
        std::float32_t scoremean;
        std::float32_t scorestddev;
        std::float32_t binstatscoremean;
        std::float32_t binstoscstddev;
        std::float32_t lead;
        std::float32_t rank;
        std::float32_t Odin;
        std::float32_t variance;
        std::float32_t estun;
        std::float32_t stddevun;
        float policies[NUM_INTERSECTIONS * POTENTIAL_MOVES];
    } m_output2;
    std::array<float, NUM_INTERSECTIONS> m_probabilities;
    std::shared_ptr<std::ofstream> m_output_stream;
    std::shared_ptr<std::ofstream> m_output_stream2;
    std::uint32_t m_version{2};
    std::int32_t m_num_samples{0};
    std::uint32_t m_compress{0};
    bool m_training{false};
};

#endif
