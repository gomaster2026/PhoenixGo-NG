#include "config.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "Utils.h"
using namespace Utils;
UCTNode::UCTNode(const int vertex, const float policy)
    : m_move(vertex), m_policy(policy) {}
bool UCTNode::first_visit() const {
    return m_visits == 0;
}
bool UCTNode::create_children(Network& network, std::atomic<int>& nodecount,
                              const GameState& state, float& eval,
                              const float min_psa_ratio) {
    if (state.get_passes() >= 2) { return false; }
    if (!acquire_expanding()) { return false; }
    if (!expandable(min_psa_ratio)) { expand_done(); return false; }
    NNCache::Netresult raw_netlist;
    try {
        raw_netlist = network.get_output(&state, Network::Ensemble::RANDOM_SYMMETRY);
    } catch (NetworkHaltException&) { expand_cancel(); throw; }
    const auto stm_eval = raw_netlist.winrate;
    const auto to_move = state.board.get_to_move();
    if (to_move == FastBoard::WHITE) {
        m_net_eval = 1.0f - stm_eval;
    } else {
        m_net_eval = stm_eval;
    }
    eval = m_net_eval;
    std::vector<Network::PolicyVertexPair> nodelist;
    auto legal_sum = 0.0f;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }
    auto pass_policy = raw_netlist.policy_pass;
    const auto relative_score_pg =
        (to_move == FastBoard::BLACK ? 1 : -1) * state.final_score();
    if (stm_eval < 0.5f && relative_score_pg < 0.0f) {
        pass_policy = std::min(pass_policy, 1e-5f);
    }
    nodelist.emplace_back(pass_policy, FastBoard::PASS);
    legal_sum += pass_policy;
    constexpr auto PG_MAX_CHILDREN_PER_NODE = 64;
    if (nodelist.size() > PG_MAX_CHILDREN_PER_NODE) {
        std::partial_sort(begin(nodelist), begin(nodelist) + PG_MAX_CHILDREN_PER_NODE, end(nodelist),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        nodelist.resize(PG_MAX_CHILDREN_PER_NODE);
        legal_sum = 0.0f;
        for (const auto& node : nodelist) { legal_sum += node.first; }
    }
    if (legal_sum > std::numeric_limits<float>::min()) {
        for (auto& node : nodelist) { node.first /= legal_sum; }
    } else {
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) { node.first = uniform_prob; }
    }
    link_nodelist(nodecount, nodelist, min_psa_ratio);
    if (first_visit()) { update(eval); }
    expand_done();
    return true;
}
void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::PolicyVertexPair>& nodelist,
                            const float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);
    if (nodelist.empty()) { return; }
    std::stable_sort(rbegin(nodelist), rend(nodelist));
    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(std::count_if(cbegin(nodelist), cend(nodelist),
            [=](const auto& node) { return node.first >= new_min_psa; }));
    } else {
        m_children.reserve(nodelist.size());
    }
    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }
    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
}
const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}
int UCTNode::get_move() const {
    return m_move;
}
void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}
void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}
void UCTNode::update(const float eval) {
    auto old_eval = static_cast<float>(m_blackevals);
    auto old_visits = static_cast<int>(m_visits);
    auto old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    m_visits++;
    accumulate_eval(eval);
    auto new_delta = eval - (old_eval + eval) / (old_visits + 1);
    auto delta = old_delta * new_delta;
    atomic_add(m_squared_eval_diff, delta);
}
bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}
bool UCTNode::expandable(const float min_psa_ratio) const {
#ifndef NDEBUG
    if (m_min_psa_ratio_children == 0.0f) {
        assert(m_expand_state.load() != ExpandState::INITIAL);
    }
#endif
    return min_psa_ratio < m_min_psa_ratio_children;
}
float UCTNode::get_policy() const {
    return m_policy;
}
void UCTNode::set_policy(const float policy) {
    m_policy = policy;
}
float UCTNode::get_eval_variance(const float default_var) const {
    return m_visits > 1 ? m_squared_eval_diff / (m_visits - 1) : default_var;
}
int UCTNode::get_visits() const {
    return m_visits;
}
float UCTNode::get_eval_lcb(const int color) const {
    auto visits = get_visits();
    if (visits < 2) { return -1e6f + visits; }
    auto mean = get_raw_eval(color);
    auto stddev = std::sqrt(get_eval_variance(1.0f) / visits);
    auto z = cached_t_quantile(visits - 1);
    return mean - z * stddev;
}
float UCTNode::get_raw_eval(const int tomove, const int virtual_loss) const {
    auto visits = get_visits() + virtual_loss;
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto eval = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        eval = 1.0f - eval;
    }
    return eval;
}
float UCTNode::get_eval(const int tomove) const {
    return get_raw_eval(tomove, m_virtual_loss);
}
float UCTNode::get_net_eval(const int tomove) const {
    if (tomove == FastBoard::WHITE) { return 1.0f - m_net_eval; }
    return m_net_eval;
}
double UCTNode::get_blackevals() const {
    return m_blackevals;
}
void UCTNode::accumulate_eval(const float eval) {
    atomic_add(m_blackevals, double(eval));
}
UCTNode* UCTNode::uct_select_child(const int color, const bool is_root) {
    wait_expanded();
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    auto total_virtual_loss = size_t{0};
    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            total_virtual_loss += static_cast<size_t>(child.get_virtual_loss());
            if (child.get_visits() > 0) {
                total_visited_policy += child.get_policy();
            }
        }
    }
    parentvisits += total_virtual_loss;
    const auto numerator = std::sqrt(std::max(double(parentvisits), 1.0));
    const auto fpu_reduction =
        (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction)
        * std::sqrt(total_visited_policy);
    const auto fpu_eval = get_raw_eval(color) - fpu_reduction;
    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();
    for (auto& child : m_children) {
        if (!child.active()) { continue; }
        auto winrate = fpu_eval;
        if (child.is_inflated()
            && child->m_expand_state.load() == ExpandState::EXPANDING) {
            winrate = -1.0f - fpu_reduction;
        } else if (child.get_visits() > 0) {
            winrate = child.get_eval(color);
        }
        const auto psa = child.get_policy();
        const auto denom = 1.0 + child.get_visits() + child.get_virtual_loss();
        const auto puct = cfg_puct * psa * (numerator / denom);
        const auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());
        if (value > best_value) {
            best_value = value;
            best = &child;
        }
    }
    assert(best != nullptr);
    best->inflate();
    return best->get();
}
class NodeComp : public std::binary_function<UCTNodePointer&, UCTNodePointer&, bool> {
public:
    NodeComp(const int color, const float lcb_min_visits)
        : m_color(color), m_lcb_min_visits(lcb_min_visits) {}
    bool operator()(const UCTNodePointer& a, const UCTNodePointer& b) {
        auto a_visit = a.get_visits();
        auto b_visit = b.get_visits();
        if (m_lcb_min_visits < 2) { m_lcb_min_visits = 2; }
        if ((a_visit > m_lcb_min_visits) && (b_visit > m_lcb_min_visits)) {
            auto a_lcb = a.get_eval_lcb(m_color);
            auto b_lcb = b.get_eval_lcb(m_color);
            if (a_lcb != b_lcb) { return a_lcb < b_lcb; }
        }
        if (a_visit != b_visit) { return a_visit < b_visit; }
        if (a_visit == 0) { return a.get_policy() < b.get_policy(); }
        return a.get_eval(m_color) < b.get_eval(m_color);
    }
private:
    int m_color;
    float m_lcb_min_visits;
};
void UCTNode::sort_children(const int color, const float lcb_min_visits) {
    std::stable_sort(rbegin(m_children), rend(m_children),
                     NodeComp(color, lcb_min_visits));
}
UCTNode& UCTNode::get_best_root_child(const int color) const {
    wait_expanded();
    assert(!m_children.empty());
    auto max_visits = 0;
    for (const auto& node : m_children) {
        max_visits = std::max(max_visits, node.get_visits());
    }
    auto ret = std::max_element(begin(m_children), end(m_children),
        NodeComp(color, cfg_lcb_min_visit_ratio * max_visits));
    ret->inflate();
    return *(ret->get());
}
size_t UCTNode::count_nodes_and_clear_expand_state() {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    if (expandable()) { m_expand_state = ExpandState::INITIAL; }
    for (auto& child : m_children) {
        if (child.is_inflated()) {
            nodecount += child->count_nodes_and_clear_expand_state();
        }
    }
    return nodecount;
}
void UCTNode::invalidate() { m_status = INVALID; }
void UCTNode::set_active(const bool active) {
    if (valid()) { m_status = active ? ACTIVE : PRUNED; }
}
bool UCTNode::valid() const { return m_status != INVALID; }
bool UCTNode::active() const { return m_status == ACTIVE; }
bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}
void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::wait_expanded() const {
    while (m_expand_state.load() == ExpandState::EXPANDING) {}
    auto v = m_expand_state.load();
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDED);
}
