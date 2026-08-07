#include "config.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "KoState.h"
#include "Random.h"
#include "UCTNode.h"
#include "Utils.h"
UCTNode* UCTNode::get_first_child() const {
    if (m_children.empty()) { return nullptr; }
    return m_children.front().get();
}
void UCTNode::kill_superkos(const GameState& state) {
    UCTNodePointer* pass_child = nullptr;
    size_t valid_count = 0;
    for (auto& child : m_children) {
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            KoState mystate = state;
            mystate.play_move(move);
            if (mystate.superko()) { child->invalidate(); }
        } else {
            pass_child = &child;
        }
        if (child->valid()) { valid_count++; }
    }
    if (valid_count > 1 && pass_child
        && !state.is_move_legal(state.get_to_move(), FastBoard::PASS)) {
        (*pass_child)->invalidate();
    }
    m_children.erase(
        std::remove_if(begin(m_children), end(m_children),
                       [](const auto& child) { return !child->valid(); }),
        end(m_children));
}
void UCTNode::dirichlet_noise(const float epsilon, const float alpha) {
    auto child_cnt = m_children.size();
    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
    }
    auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);
    if (sample_sum < std::numeric_limits<float>::min()) { return; }
    for (auto& v : dirichlet_vector) { v /= sample_sum; }
    child_cnt = 0;
    for (auto& child : m_children) {
        auto policy = child->get_policy();
        auto eta_a = dirichlet_vector[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        child->set_policy(policy);
    }
}
void UCTNode::randomize_first_proportionally() {
    auto accum = 0.0;
    auto norm_factor = 0.0;
    auto accum_vector = std::vector<double>{};
    for (const auto& child : m_children) {
        auto visits = child->get_visits();
        if (norm_factor == 0.0) {
            norm_factor = visits;
            if (visits <= cfg_random_min_visits) { return; }
        }
        if (visits > cfg_random_min_visits) {
            accum += std::pow(visits / norm_factor, 1.0 / cfg_random_temp);
            accum_vector.emplace_back(accum);
        }
    }
    auto distribution = std::uniform_real_distribution<double>{0.0, accum};
    auto pick = distribution(Random::get_Rng());
    auto index = size_t{0};
    for (size_t i = 0; i < accum_vector.size(); i++) {
        if (pick < accum_vector[i]) { index = i; break; }
    }
    if (index == 0) { return; }
    assert(m_children.size() > index);
    std::iter_swap(begin(m_children), begin(m_children) + index);
}
UCTNode* UCTNode::get_nopass_child(FastState& state) const {
    for (const auto& child : m_children) {
        if (child->m_move != FastBoard::PASS
            && !state.board.is_eye(state.get_to_move(), child->m_move)) {
            return child.get();
        }
    }
    return nullptr;
}
std::unique_ptr<UCTNode> UCTNode::find_child(const int move) {
    for (auto& child : m_children) {
        if (child.get_move() == move) {
            child.inflate();
            return std::unique_ptr<UCTNode>(child.release());
        }
    }
    return nullptr;
}
void UCTNode::inflate_all_children() {
    for (const auto& node : get_children()) { node.inflate(); }
}
void UCTNode::prepare_root_node(Network& network, const int color,
                                std::atomic<int>& nodes, GameState& root_state) {
    float root_eval;
    const auto had_children = has_children();
    if (expandable()) {
        create_children(network, nodes, root_state, root_eval);
    }
    if (had_children) {
        root_eval = get_net_eval(color);
    } else {
        root_eval = (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval);
    }
    Utils::myprintf("NN eval=%f\n", root_eval);
    inflate_all_children();
    kill_superkos(root_state);
    {
        const float epsilon = cfg_noise ? 0.25f : 0.1f;
        const float alpha = (cfg_noise ? 0.03f : 0.01f) * 361.0f / NUM_INTERSECTIONS;
        dirichlet_noise(epsilon, alpha);
    }
}
