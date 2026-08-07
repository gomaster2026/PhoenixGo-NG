/* SMP.cpp - Simple spinlock mutex */
#include <cassert>
#include <thread>
#include "SMP.h"
SMP::Mutex::Mutex() {
    m_lock = false;
}
SMP::Lock::Lock(Mutex& m) {
    m_mutex = &m;
    lock();
}
void SMP::Lock::lock() {
    assert(!m_owns_lock);
    while (m_mutex->m_lock.exchange(true, std::memory_order_acquire)) {
        while (m_mutex->m_lock.load(std::memory_order_relaxed)) {}
    }
    m_owns_lock = true;
}
void SMP::Lock::unlock() {
    assert(m_owns_lock);
    auto lock_held = m_mutex->m_lock.exchange(false, std::memory_order_release);
#ifdef NDEBUG
    (void)lock_held;
#else
    assert(lock_held);
#endif
    m_owns_lock = false;
}
SMP::Lock::~Lock() {
    if (m_owns_lock) {
        unlock();
    }
}
size_t SMP::get_num_cpus() {
    return std::thread::hardware_concurrency();
}
