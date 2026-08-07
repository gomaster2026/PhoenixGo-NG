#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"
#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <vector>
#include "GameState.h"
#include "SMP.h"
#include "Utils.h"
#include "ThreadPool.h"
#include "UCTNodePointer.h"

class UCTNode {
public:
    using sorted_node_pool_iterator = std::vector<UCTNodePointer>::const_iterator;
    using node_pool_t = std::vector<UCTNodePointer>;
    using edge_iterator = std::vector<UCTNodePointer>::const_iterator;
    struct NodePolicyData {
        std::uint16_t m_policy;
        std::uint16_t m_prev_move_stats;
        float m_net_eval;
        float m_init_prob;
        float m_risk_penalty;
    };
    struct NodeCacheData {
        std::uint32_t m_cache_hash;
        std::uint16_t m_lower_bound;
        std::uint16_t m_upper_bound;
        std::uint32_t m_visits;
    };
    struct NodeMutexData {
        std::atomic<bool> m_has_children{false};
        std::atomic<bool> m_is_prepended{false};
    };
    struct NodeVirtualLoss {
        static constexpr auto VIRTUAL_LOSS_COUNT = 1;
        std::atomic<int> m_virtual_losses{0};
    };
    UCTNode() = delete;
    UCTNode(int vertex, float score, float init_prob);
    ~UCTNode() = default;
    UCTNode(int vertex, float score, float init_prob, float eval);
    UCTNode(const UCTNode&) = delete;
    UCTNode& operator=(const UCTNode&) = delete;
    int get_vertex() const;
    std::uint16_t get_policy() const;
    std::uint16_t get_prev_move_stats() const;
    void set_policy(int p);
    void set_prev_move_stats(int p);
    bool first_visit() const;
    bool has_children() const;
    bool has_netsibling() const;
    bool is_move_pass(int move) const;
    bool is_valid() const;
    bool get_renforce(int color, int vertex, bool* kill) const;
    bool get_selfrepeat(int color, int vertex) const;
    void kill();
    bool must_visits(int* vertex, int *last_move, int color) const;
    bool virtual_loss_affected() const { return false; }
    void add_starting_pol(int policy) { m_node.m_policy = policy; }
    std::uint16_t get_starting_pol() const { return m_node.m_policy; }
    void virtual_loss(bool is_resign = false);
    void virtual_loss_undo();
    bool active() const;
    void accumulate_eval(float eval, float* eval_averager) const;
    float get_value(float default_value) const;
    void set_active(bool active, bool up_to_root = false);
    std::string get_name() const;
    UCTNode* parent_node();
    const UCTNode* parent_node() const;
    bool is_root() const;
    bool is_sub_root() const;
    bool is_top_move() const;
    int get_root_move() const;
    std::uint32_t get_visits() const;
    void set_visits(int visits);
    std::uint64_t get_cummulativechildNodesVisits() const;
    std::uint32_t get_childsVisits() const;
    void set_childsVisits(std::uint32_t visits);
    bool move_VISITS() const { return m_node_type == 1; }
    bool move_COWARDS() const { return m_node_type == 2; }
    bool move_PRIMED() const { return m_node_type == 3; }
    bool move_PASS() const { return m_vertex == FastBoard::PASS; }
    float get_risk_penalty() const { return m_node.m_risk_penalty; }
    float get_eval() const;
    float get_rscore() const;
    float get_score() const;
    void set_risk_penalty(float risk_penalty) { m_node.m_risk_penalty = risk_penalty; }
    void set_eval(float eval) { m_node.m_net_eval = eval; }
    float get_rsize() const { return m_rsize; }
    void set_rsize(float rsize) { m_rsize = rsize; }
    bool is_result_eval() const { return m_is_result_eval; }
    bool get_state_eval() const { return m_state_eval; }
    bool get_is_ladder() const { return m_is_ladder; }
    float get_pseudo_cost() const { return m_pseudo_cost; }
    void set_state_eval(bool stateeval) { m_state_eval = stateeval; }
    float get_pseudo_cost_fast() const { return m_pseudo_cost_fast; }
    void set_pseudo_cost_fast(float v) { m_pseudo_cost_fast = v; }
    float get_pseudo_cost_slow() const { return m_pseudo_cost_slow; }
    void set_pseudo_cost_slow(float v) { m_pseudo_cost_slow = v; }
    float get_maxdepth() const { return m_maxdepth; }
    float get_cost() const { return m_cost; }
    void set_cost(float cost) { m_cost = cost; }
    void set_maxdepth(float d) { m_maxdepth = d; }
    float get_scoremean() const { return m_scoremean; }
    void set_scoremean(float v) { m_scoremean = v; }
    float get_scorestddev() const { return m_scorestddev; }
    void set_scorestddev(float v) { m_scorestddev = v; }
    float get_lead() const { return m_lead; }
    void set_lead(float v) { m_lead = v; }
    std::uint16_t get_lcb_count() const { return m_lcb_count; }
    void set_lcb_count(std::uint16_t v) { m_lcb_count = v; }
    std::uint16_t get_rave_childcount() const { return m_rave_childcount; }
    void set_rave_childcount(std::uint16_t v) { m_rave_childcount = v; }
    std::uint16_t get_rave_count() const { return m_rave_count; }
    void set_rave_count(std::uint16_t v) { m_rave_count = v; }
    float get_rave_X() const { return m_rave_X; }
    void set_rave_X(float v) { m_rave_X = v; }
    float get_rave_Y() const { return m_rave_Y; }
    void set_rave_Y(float v) { m_rave_Y = v; }
    float get_lcb() const { return m_lcb; }
    void set_lcb(float v) { m_lcb = v; }
    bool get_needchildstats() const { return m_needchildstats; }
    void set_needchildstats(bool v) { m_needchildstats = v; }
    bool is_ladder() const { return m_is_ladder; }
    void set_is_ladder(bool v) { m_is_ladder = v; }
    void set_bestmove(bool v) { m_bestmove = v; }
    bool is_bestmove() const { return m_bestmove; }
    void set_renforce(int v) { m_renforce = v; }
    void set_renforce2(int v) { m_renforce2 = v; }
    int get_renforce() const { return m_renforce; }
    int get_renforce2() const { return m_renforce2; }
    bool need_renforce() const { return m_renforce > 0 && m_renforce2 <= 0; }
    void set_needrenforce(bool v) { m_needrenforce = v; }
    bool get_needrenforce() const { return m_needrenforce; }
    bool get_needrenforce_fast() const { return m_needrenforce_fast; }
    void set_needrenforce_fast(bool v) { m_needrenforce_fast = v; }
    bool get_needrenforce_slow() const { return m_needrenforce_slow; }
    void set_needrenforce_slow(bool v) { m_needrenforce_slow = v; }
    float get_eval_stab() const { return m_eval_stab; }
    void set_eval_stab(float v) { m_eval_stab = v; }
    std::uint16_t get_max_visits() const { return m_max_visits; }
    void set_max_visits(std::uint16_t v) { m_max_visits = v; }
    void set_is_result_eval(bool v) { m_is_result_eval = v; }
    float get_best_child_weight() const { return m_best_child_weight; }
    void set_best_child_weight(float v) { m_best_child_weight = v; }
    float get_best_child_eval() const { return m_best_child_eval; }
    void set_best_child_eval(float v) { m_best_child_eval = v; }
    std::uint32_t get_best_child_visits() const { return m_best_child_visits; }
    void set_best_child_visits(std::uint32_t v) { m_best_child_visits = v; }
    int get_best_child_vertex() const { return m_best_child_vertex; }
    void set_best_child_vertex(int v) { m_best_child_vertex = v; }
    void increment_visit_count() { m_visits += 1; }
    void add_to_childrensVisits(int increment) { m_childrensVisits += increment; }
    std::uint32_t get_selfVisits() const { return m_selfVisits; }
    void set_selfVisits(std::uint32_t v) { m_selfVisits = v; }
    void set_node_type(int nodetype) { m_node_type = nodetype; }
    std::uint16_t get_bestmove_count() const { return m_bestmove_count; }
    void set_bestmove_count(std::uint16_t v) { m_bestmove_count = v; }
    std::uint16_t get_rsize_count() const { return m_rsize_count; }
    void set_rsize_count(std::uint16_t v) { m_rsize_count = v; }
    std::uint16_t get_ladder_count() const { return m_ladder_count; }
    void set_ladder_count(std::uint16_t v) { m_ladder_count = v; }
    void sort_policies(node_pool_t* node_pool);
    void sort_children(node_pool_t* node_pool);
    void cap_best_child(float* eval_averager);
    float accumulate_risk(node_pool_t* node_pool, float eval_averager, int color) const;
    bool have_children() const { return m_children != nullptr; }
    bool expandable() const { return m_expandable; }
    float get_policy() const { return m_node.m_init_prob; }
    float get_net_eval() const { return m_node.m_net_eval; }
    std::uint32_t get_CacheHash() const { return m_node.m_cache_hash; }
    std::uint16_t get_lowerbound() const { return m_node.m_lower_bound; }
    std::uint16_t get_upperbound() const { return m_node.m_upper_bound; }
    void set_CacheHash(std::uint32_t hash) { m_node.m_cache_hash = hash; }
    void set_lowerbound(std::uint16_t v) { m_node.m_lower_bound = v; }
    void set_upperbound(std::uint16_t v) { m_node.m_upper_bound = v; }
    std::uint32_t get_cache_policy() const { return m_node.m_policy; }
    void set_cache_policy(std::uint32_t policy) { m_node.m_policy = policy; }
private:
    void link_node(UCTNode* child);
    void sort_child(node_pool_t* node_pool);
    bool m_expandable{true};
    std::uint16_t m_max_visits{0};
    bool m_bestmove{false};
    bool m_is_ladder{false};
    int m_renforce{0};
    int m_renforce2{0};
    bool m_needrenforce{false};
    bool m_needrenforce_fast{false};
    bool m_needrenforce_slow{false};
    float m_best_child_eval{0.0f};
    float m_best_child_weight{0.0f};
    std::uint32_t m_best_child_visits{0};
    int m_best_child_vertex{-1};
    float m_eval_stab{0.0f};
    float m_pseudo_cost{0.0f};
    float m_pseudo_cost_fast{0.0f};
    float m_pseudo_cost_slow{0.0f};
    float m_rsize{0.0f};
    bool m_is_result_eval{false};
    bool m_state_eval{false};
    std::uint32_t m_CacheHash{0};
    std::uint16_t m_lcb_count{0};
    std::uint16_t m_rave_childcount{0};
    std::uint16_t m_rave_count{0};
    float m_rave_X{0.0f};
    float m_rave_Y{0.0f};
    float m_lcb{0.0f};
    bool m_needchildstats{false};
    std::uint16_t m_bestmove_count{0};
    std::uint16_t m_rsize_count{0};
    std::uint16_t m_ladder_count{0};
    std::uint32_t m_childrensVisits{0};
    std::uint32_t m_selfVisits{0};
    int m_node_type{0};
    float m_cost{0.0f};
    float m_maxdepth{0.0f};
    float m_scoremean{0.0f};
    float m_scorestddev{0.0f};
    float m_lead{0.0f};
    std::atomic<bool> m_is_active{false};
    std::uint32_t m_visits{0};
    int m_vertex;
    NodePolicyData m_node;
    NodeCacheData m_cached_node;
    NodeMutexData m_has_children_mutex;
    NodeVirtualLoss m_vloss_node;
    using node_ptr_t = std::unique_ptr<UCTNode>;
    std::unique_ptr<node_pool_t> m_children;
};

#endif
