#include "threadpool.h"
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#if !defined(_DISABLE_PTHREAD)
#if defined(__has_include)
#if __has_include(<pthread.h>)
#include <pthread.h>
#define _HAVE_PTHREAD 1
#endif
#elif defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#define _HAVE_PTHREAD 1
#endif
#endif

typedef struct TaskNode {
	void (*fn)(void*);
	void* arg;
	struct TaskNode* next;
} TaskNode;

struct ThreadPool {
#if defined(_HAVE_PTHREAD)
	pthread_t* workers;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	pthread_cond_t idle;
#endif
	size_t num_workers;
	TaskNode* head;
	TaskNode* tail;
	int stop;
	int active;
};

static int hardware_concurrency(void) {
#if defined(_WIN32)
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return (int)info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	return (n > 0) ? (int)n : 4;
#else
	return 4;
#endif
}

#if defined(_HAVE_PTHREAD)
static void* worker_loop(void* arg) {
	ThreadPool* pool = (ThreadPool*)arg;
	for (;;) {
		pthread_mutex_lock(&pool->mutex);
		while (!pool->head && !pool->stop) {
			pthread_cond_wait(&pool->cond, &pool->mutex);
		}
		if (pool->stop && !pool->head) {
			pthread_mutex_unlock(&pool->mutex);
			break;
		}
		TaskNode* task = pool->head;
		if (task) {
			pool->head = task->next;
			if (!pool->head) pool->tail = NULL;
			pool->active++;
		}
		pthread_mutex_unlock(&pool->mutex);

		if (task) {
			task->fn(task->arg);
			free(task);
			pthread_mutex_lock(&pool->mutex);
			pool->active--;
			if (!pool->head && pool->active == 0) {
				pthread_cond_broadcast(&pool->idle);
			}
			pthread_mutex_unlock(&pool->mutex);
		}
	}
	return NULL;
}
#endif

ThreadPool* threadpool_create(size_t num_threads) {
	if (num_threads == 0) num_threads = (size_t)hardware_concurrency();
	if (num_threads == 0) num_threads = 1;

	ThreadPool* pool = (ThreadPool*)calloc(1, sizeof(ThreadPool));
	if (!pool) return NULL;
	pool->num_workers = num_threads;

#if defined(_HAVE_PTHREAD)
	pool->workers = (pthread_t*)calloc(num_threads, sizeof(pthread_t));
	pthread_mutex_init(&pool->mutex, NULL);
	pthread_cond_init(&pool->cond, NULL);
	pthread_cond_init(&pool->idle, NULL);
	for (size_t i = 0; i < num_threads; i++) {
		pthread_create(&pool->workers[i], NULL, worker_loop, pool);
	}
#endif
	return pool;
}

void threadpool_destroy(ThreadPool* pool) {
	if (!pool) return;
#if defined(_HAVE_PTHREAD)
	pthread_mutex_lock(&pool->mutex);
	pool->stop = 1;
	pthread_cond_broadcast(&pool->cond);
	pthread_mutex_unlock(&pool->mutex);
	for (size_t i = 0; i < pool->num_workers; i++) {
		pthread_join(pool->workers[i], NULL);
	}
	pthread_mutex_destroy(&pool->mutex);
	pthread_cond_destroy(&pool->cond);
	pthread_cond_destroy(&pool->idle);
	free(pool->workers);
#endif
	while (pool->head) {
		TaskNode* n = pool->head;
		pool->head = n->next;
		free(n);
	}
	free(pool);
}

size_t threadpool_size(const ThreadPool* pool) {
	return pool ? pool->num_workers : 0;
}

void threadpool_enqueue(ThreadPool* pool, void (*fn)(void*), void* arg) {
	if (!pool || !fn) return;

#if defined(_HAVE_PTHREAD)
	TaskNode* task = (TaskNode*)malloc(sizeof(TaskNode));
	if (!task) return;
	task->fn = fn;
	task->arg = arg;
	task->next = NULL;

	pthread_mutex_lock(&pool->mutex);
	if (pool->stop) {
		pthread_mutex_unlock(&pool->mutex);
		free(task);
		return;
	}
	if (!pool->tail) pool->head = task;
	else pool->tail->next = task;
	pool->tail = task;
	pthread_cond_signal(&pool->cond);
	pthread_mutex_unlock(&pool->mutex);
#else
	fn(arg);
#endif
}

static ThreadPool* g_pool = NULL;

ThreadPool* get_global_thread_pool(void) {
	if (!g_pool) g_pool = threadpool_create(0);
	return g_pool;
}

typedef struct {
	parallel_for_fn fn;
	void* ctx;
	int start;
	int end;
} range_task_ctx;

static void range_task(void* arg) {
	range_task_ctx* r = (range_task_ctx*)arg;
	for (int i = r->start; i < r->end; i++) {
		r->fn(i, r->ctx);
	}
}

void threadpool_parallel_for(int start, int end, int num_threads, parallel_for_fn fn, void* ctx) {
	if (!fn || end <= start) return;

	int total = end - start;
	if (num_threads <= 1 || total <= num_threads) {
		for (int i = start; i < end; i++) fn(i, ctx);
		return;
	}

	ThreadPool* pool = get_global_thread_pool();
	if (!pool) {
		for (int i = start; i < end; i++) fn(i, ctx);
		return;
	}

	int work_per = total / num_threads;
	int rem = total % num_threads;
	int cur = start;
	range_task_ctx* tasks = (range_task_ctx*)calloc((size_t)num_threads, sizeof(range_task_ctx));
	if (!tasks) {
		for (int i = start; i < end; i++) fn(i, ctx);
		return;
	}

#if defined(_HAVE_PTHREAD)
	pthread_mutex_lock(&pool->mutex);
#endif
	for (int t = 0; t < num_threads; t++) {
		int chunk = work_per + (t < rem ? 1 : 0);
		tasks[t].fn = fn;
		tasks[t].ctx = ctx;
		tasks[t].start = cur;
		tasks[t].end = cur + chunk;
		cur += chunk;
		threadpool_enqueue(pool, range_task, &tasks[t]);
	}

#if defined(_HAVE_PTHREAD)
	while (pool->head || pool->active > 0) {
		pthread_cond_wait(&pool->idle, &pool->mutex);
	}
	pthread_mutex_unlock(&pool->mutex);
#endif

	free(tasks);
}
