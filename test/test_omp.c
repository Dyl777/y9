#include <stdio.h>
#include <omp.h>

int main() {
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    
    int thread_counts[16] = {0};
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 1000; i++) {
        int tid = omp_get_thread_num();
        if (tid < 16) thread_counts[tid]++;
    }
    
    printf("Thread distribution:\n");
    int active = 0;
    for (int i = 0; i < 16; i++) {
        if (thread_counts[i] > 0) {
            printf("  Thread %d: %d iterations\n", i, thread_counts[i]);
            active++;
        }
    }
    printf("Total threads used: %d\n", active);
    return 0;
}
