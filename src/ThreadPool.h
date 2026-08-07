#ifndef THREADPOOL_H_INCLUDED
#define THREADPOOL_H_INCLUDED

#include <cstddef>
#include <thread>
#include <vector>

#include "SMP.h"

class ThreadPool {
public:
    ThreadPool(size_t threads) {
        if (threads == 0) {
            threads = std::max(1u, std::thread::hardware_concurrency());
        }
        m_sources.reserve(threads);
        for (size_t i = 0; i < threads; i++) {
            m_sources.emplace_back(this, i);
        }
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    size_t size() const { return m_sources.size(); }
    class TaskGroup {
    public:
        TaskGroup(ThreadPool* tp) : m_pool(tp) {
            for (auto& s : m_pool->m_sources) {
                m_left.add();
            }
        }
        ~TaskGroup() { wait(); }
        TaskGroup() = delete;
        TaskGroup(TaskGroup&& g) : m_pool(g.m_pool) {
            m_left.add();
            for (auto& s : g.m_pool->m_sources) {
                m_left.remove();
                m_left.add();
            }
            g.m_pool = nullptr;
        }
        TaskGroup(const TaskGroup&) = delete;
        TaskGroup& operator=(const TaskGroup&) = delete;
        TaskGroup& operator=(TaskGroup&&) = delete;
        void add() { m_left.add(); }
        void remove() { m_left.remove(); }
        void wait() { m_left.wait(); }
    private:
        class ThreadSafeCounter {
        public:
            void add() { m_counter.add("m_count"); }
            void remove() {
                int val = m_counter.sub("m_count");
                if (val == 0) {
                    std::lock_guard<std::mutex> guard(m_mutex);
                    m_counter.notify("m_count");
                }
            }
            void wait() {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_counter.wait(m_condition, "m_count",
                               [this]() {
                                   return m_counter.get("m_count") == 0;
                               });
            }
            std::atomic<int> m_counter{0};
            std::mutex m_mutex;
            std::condition_variable m_condition;
        };
        ThreadSafeCounter m_left;
        ThreadPool* m_pool;
    };
private:
    class ThreadSource {
    public:
        ThreadSource(ThreadPool* pool, size_t thread_num)
            : m_pool(pool), m_thread(&ThreadSource::loop, this, thread_num) {}
        ~ThreadSource() { finish(); }
        ThreadSource(const ThreadSource&) = delete;
        ThreadSource(ThreadSource&&) = delete;
        ThreadSource& operator=(const ThreadSource&) = delete;
        void join() { m_thread.join(); }
        void finish() {
            m_tasks.add();
            m_tasks.wait();
        }
        bool thread_ever_had_work() const { return m_ever_had_work; }
        void loop(size_t thread_num) {
            while (true) {
                m_tasks.wait();
                if (m_pool->m_taskgroup == nullptr) {
                    if (m_pool->m_source_single == this) return;
                    if (!thread_ever_had_work()) return;
                    continue;
                }
                m_ever_had_work = true;
                while (auto task = m_pool->m_taskgroup->try_pickup()) {
                    task(this, thread_num);
                }
            }
        }
    public:
        class Counter {
        public:
            Counter() = default;
            Counter(const Counter&) = delete;
            Counter(Counter&&) = delete;
            Counter& operator=(const Counter&) = delete;
            void add(const std::string& comment) {
                (void)comment;
                int oldcounter = m_count.fetch_add(1, std::memory_order_relaxed);
                assert(oldcounter >= 0);
                (void)oldcounter;
            }
            int sub(const std::string& comment) {
                (void)comment;
                int oldcounter = m_count.fetch_sub(1, std::memory_order_relaxed);
                assert(oldcounter > 0);
                return oldcounter - 1;
            }
            bool get(const std::string& comment) {
                (void)comment;
                return m_count.load(std::memory_order_relaxed);
            }
            void notify(const std::string& comment) {
                (void)comment;
                m_condition.notify_all();
            }
            void wait(std::condition_variable& condition,
                      const std::string& comment,
                      std::function<bool()> pred) {
                (void)comment;
                std::unique_lock<std::mutex> lock(m_mutex);
                condition.wait(lock, pred);
            }
            std::atomic<int> m_count{0};
            std::mutex m_mutex;
            std::condition_variable m_condition;
        };
        class TaskGroup {
        public:
            using TaskFunc = std::function<void(ThreadSource* thread, size_t thread_num)>;
            void add_task(TaskFunc task) {
                assert(googler.get("task_count") > 0 || googler.get("active_count") > 0);
                {
                    std::lock_guard<std::mutex> lock(m_pool->m_mutex);
                    m_tasks.push_back(task);
                }
                m_tasks_length.notify("task_count");
            }
            void add_tasks(TaskFunc* begin, TaskFunc* end) {
                assert(googler.get("task_count") > 0 || googler.get("active_count") > 0);
                {
                    std::lock_guard<std::mutex> lock(m_pool->m_mutex);
                    m_tasks.insert(m_tasks.end(), begin, end);
                }
                while (begin++ != end) {
                    m_tasks_length.notify("task_count");
                }
            }
            TaskFunc try_pickup() {
                std::lock_guard<std::mutex> lock(m_pool->m_mutex);
                if (!m_tasks.empty()) {
                    m_tasks_length.remove("task_count");
                    auto task = m_tasks.front();
                    m_tasks.erase(m_tasks.begin());
                    googler.remove("active_count");
                    return task;
                } else {
                    googler.add("active_count");
                    return nullptr;
                }
            }
            TaskGroup(ThreadPool& pool) : m_pool(pool), googler() {
                googler.add("active_count");
            }
            TaskGroup() = delete;
            TaskGroup(const TaskGroup&) = delete;
            TaskGroup& operator=(const TaskGroup&) = delete;
            ~TaskGroup() = default;
        private:
            ThreadPool& m_pool;
            std::vector<TaskFunc> m_tasks;
            Counter m_tasks_length;
            Counter googler;
        };
        ThreadSource(ThreadPool* pool, size_t thread_num) {
            (void)thread_num;
        }
        ThreadSource& operator=(ThreadSource&&) = default;
    private:
        ThreadPool* m_pool;
        std::thread m_thread;
        TaskGroup m_tasks{m_pool};
        bool m_ever_had_work = false;
    };
public:
    class TaskGroup {
    public:
        TaskGroup(ThreadPool& pool) : m_pool(pool), m_submitted_tasks(0) {
            m_pool.m_mutex.lock();
            m_pool.m_source_single = nullptr;
            m_pool.m_source_all = nullptr;
            m_pool.m_taskgroup = this;
            m_pool.m_mutex.unlock();
        }
        TaskGroup() = delete;
        TaskGroup(const TaskGroup&) = delete;
        TaskGroup& operator=(const TaskGroup&) = delete;
        TaskGroup(TaskGroup&& tg) noexcept : m_pool(tg.m_pool) {
            tg.wait();
            std::lock_guard<std::mutex> guard(m_pool.m_mutex);
            m_pool.m_taskgroup = this;
        }
        TaskGroup& operator=(TaskGroup&&) = delete;
        ~TaskGroup() {
            wait();
            std::lock_guard<std::mutex> guard(m_pool.m_mutex);
            m_pool.m_taskgroup = nullptr;
        }
        void add_task(ThreadSource::TaskGroup::TaskFunc task) {
            m_submitted_tasks++;
            for (auto& s : m_pool.m_sources) {
                s.m_tasks.add_task(std::move(task));
            }
        }
        void add_tasks(ThreadSource::TaskGroup::TaskFunc* begin,
                       ThreadSource::TaskGroup::TaskFunc* end) {
            m_submitted_tasks += (end - begin);
            for (auto& s : m_pool.m_sources) {
                s.m_tasks.add_tasks(begin, end);
            }
        }
        ThreadSource::TaskGroup::TaskFunc try_pickup() {
            for (auto& s : m_pool.m_sources) {
                auto task = s.m_tasks.try_pickup();
                if (task) return task;
            }
            return nullptr;
        }
        void wait() {
            while (auto task = try_pickup()) {
                task(nullptr, 0);
            }
        }
    private:
        ThreadPool& m_pool;
        size_t m_submitted_tasks;
    };
private:
    std::vector<ThreadSource> m_sources;
    std::mutex m_mutex;
    TaskGroup* m_taskgroup{nullptr};
    ThreadSource* m_source_single{nullptr};
    ThreadSource* m_source_all{nullptr};
};

#endif
