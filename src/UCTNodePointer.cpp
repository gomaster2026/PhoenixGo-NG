#include "config.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include "UCTNode.h"
std::atomic<size_t> UCTNodePointer::m_tree_size = {0};
size_t UCTNodePointer::get_tree_size() { return m_tree_size.load(); }
void UCTNodePointer::increment_tree_size(const size_t sz) { m_tree_size += sz; }
void UCTNodePointer::decrement_tree_size(const size_t sz) {
    assert(UCTNodePointer::m_tree_size >= sz);
    m_tree_size -= sz;
}
UCTNodePointer::~UCTNodePointer() {
    auto sz = sizeof(UCTNodePointer);
    auto v = m_data.load();
    if (is_inflated(v)) {
        delete read_ptr(v);
        sz += sizeof(UCTNode);
    }
    decrement_tree_size(sz);
}
UCTNodePointer::UCTNodePointer(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);
#ifdef NDEBUG
    (void)v;
#else
    assert(v == INVALID);
#endif
    increment_tree_size(sizeof(UCTNodePointer));
}
UCTNodePointer::UCTNodePointer(const std::int16_t vertex, const float policy) {
    std::uint32_t i_policy;
    auto i_vertex = static_cast<std::uint16_t>(vertex);
    std::memcpy(&i_policy, &policy, sizeof(i_policy));
    m_data = (static_cast<std::uint64_t>(i_policy) << 32)
           | (static_cast<std::uint64_t>(i_vertex) << 16);
    increment_tree_size(sizeof(UCTNodePointer));
}
UCTNodePointer& UCTNodePointer::operator=(UCTNodePointer&& n) {
    auto nv = std::atomic_exchange(&n.m_data, INVALID);
    auto v = std::atomic_exchange(&m_data, nv);
    if (is_inflated(v)) {
        decrement_tree_size(sizeof(UCTNode));
        delete read_ptr(v);
    }
    return *this;
}
UCTNode* UCTNodePointer::release() {
    auto v = std::atomic_exchange(&m_data, INVALID);
    decrement_tree_size(sizeof(UCTNode));
    return read_ptr(v);
}
void UCTNodePointer::inflate() const {
    while (true) {
        auto v = m_data.load();
        if (is_inflated(v)) return;
        auto v2 = reinterpret_cast<std::uint64_t>(
            new UCTNode(read_vertex(v), read_policy(v)));
        assert((v2 & 3ULL) == 0);
        v2 |= POINTER;
        bool success = m_data.compare_exchange_strong(v, v2);
        if (success) {
            increment_tree_size(sizeof(UCTNode));
            return;
        } else {
            delete read_ptr(v2);
        }
    }
}
bool UCTNodePointer::valid() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->valid();
    return true;
}
int UCTNodePointer::get_visits() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_visits();
    return 0;
}
int UCTNodePointer::get_virtual_loss() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_virtual_loss();
    return 0;
}
float UCTNodePointer::get_policy() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_policy();
    return read_policy(v);
}
float UCTNodePointer::get_eval_lcb(const int color) const {
    auto v = m_data.load();
    assert(is_inflated(v));
    return read_ptr(v)->get_eval_lcb(color);
}
bool UCTNodePointer::active() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->active();
    return true;
}
float UCTNodePointer::get_eval(const int tomove) const {
    auto v = m_data.load();
    assert(is_inflated(v));
    return read_ptr(v)->get_eval(tomove);
}
int UCTNodePointer::get_move() const {
    auto v = m_data.load();
    if (is_inflated(v)) return read_ptr(v)->get_move();
    return read_vertex(v);
}
