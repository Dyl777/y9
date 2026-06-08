#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <stddef.h>
#include <stdlib.h>

#if defined(_OPENMP) && !defined(_DISABLE_OPENMP)
#include <omp.h>
#endif

typedef void (*parallel_for_fn)(int i, void* ctx);

typedef struct ThreadPool ThreadPool;

ThreadPool* threadpool_create(size_t num_threads);
void threadpool_destroy(ThreadPool* pool);
size_t threadpool_size(const ThreadPool* pool);
void threadpool_enqueue(ThreadPool* pool, void (*fn)(void*), void* arg);
ThreadPool* get_global_thread_pool(void);
void threadpool_parallel_for(int start, int end, int num_threads, parallel_for_fn fn, void* ctx);

static inline int threadpool_default_threads(void) {
#if defined(_OPENMP) && !defined(_DISABLE_OPENMP)
	int n = omp_get_max_threads();
	return (n > 0) ? n : 1;
#else
	return 1;
#endif
}

static inline void parallel_for(int start, int end, int num_threads, parallel_for_fn fn, void* ctx) {
	if (!fn || end <= start) return;

	int total = end - start;
	if (num_threads <= 0) num_threads = threadpool_default_threads();
	if (num_threads <= 1 || total <= num_threads) {
		for (int i = start; i < end; i++) fn(i, ctx);
		return;
	}

	threadpool_parallel_for(start, end, num_threads, fn, ctx);
}



/*IMPLEMENT ALL THE FEATURES OF THIS THREADPOOL IN C HERE ADD IT DON'T DELETEE ANYTHING
#ifndef JNET_CORE_THREADPOOL_H
#define JNET_CORE_THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

namespace JNet {

class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    // Submit a task to the thread pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // Get number of threads
    size_t size() const { return workers.size(); }

    // Disable copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

// Global thread pool instance
extern ThreadPool& getGlobalThreadPool();

// Utility function to parallelize loops
template<typename Func>
void parallel_for(size_t start, size_t end, size_t num_threads, Func&& func) {
    if (end <= start) return;
    
    size_t total_work = end - start;
    if (total_work <= num_threads || num_threads <= 1) {
        // Not worth parallelizing or single thread requested
        for (size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    ThreadPool& pool = getGlobalThreadPool();
    std::vector<std::future<void>> futures;
    
    size_t work_per_thread = total_work / num_threads;
    size_t remaining_work = total_work % num_threads;
    
    size_t current_start = start;
    
    for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
        size_t thread_work = work_per_thread + (thread_id < remaining_work ? 1 : 0);
        size_t current_end = current_start + thread_work;
        
        futures.emplace_back(
            pool.enqueue([func, current_start, current_end]() {
                for (size_t i = current_start; i < current_end; ++i) {
                    func(i);
                }
            })
        );
        
        current_start = current_end;
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

// Template implementation
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if(stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

}

#endif // JNET_CORE_THREADPOOL_H
*/

#endif
