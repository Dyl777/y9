#include <stdio.h>
#include <omp.h>
#include <time.h>

#define SEQ_LEN 16000
#define EMB_DIM 32

int main() {
    printf("Max OpenMP threads: %d\n\n", omp_get_max_threads());
    
    float data[SEQ_LEN * EMB_DIM];
    for (int i = 0; i < SEQ_LEN * EMB_DIM; i++) data[i] = (float)i;
    
    volatile float sum = 0;
    clock_t start = clock();
    
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int i = 0; i < SEQ_LEN; i++) {
        if (i == 0) {
            printf("Thread %d processing rows starting at %d\n", 
                   omp_get_thread_num(), i);
        }
        float row_sum = 0;
        for (int j = 0; j < EMB_DIM; j++) {
            row_sum += data[i * EMB_DIM + j];
        }
        sum += row_sum;
    }
    
    clock_t end = clock();
    printf("\nSum: %.0f, Time: %.3f ms\n", sum, 
           (double)(end - start) * 1000.0 / CLOCKS_PER_SEC);
    return 0;
}
