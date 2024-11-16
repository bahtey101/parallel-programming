#include <omp.h>
#include <stdio.h>

#define NMAX         1000
#define ITERS_NUMBER 20
#define Q            1

double sequental_time(int* a, int* b) {
    double sum = 0, start_time, end_time, total_time = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = omp_get_wtime();
        for (i = 0; i < NMAX; ++i) {
            for (j = 0; j < Q; ++j) {
                sum = sum + a[i] + b[i];
            }
        }
        end_time = omp_get_wtime();
        total_time = total_time + (end_time - start_time);
    }
    sum = sum / (ITERS_NUMBER * Q);
    printf("\nSQUENTAL SUM  : %.0f", sum);

    return total_time / ITERS_NUMBER;
}

double reduction_time(int* a, int* b) {
    double sum = 0, start_time, end_time, total_time = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = omp_get_wtime(); 

#pragma omp parallel for reduction(+:sum) shared(a, b) private(i, j)
        for (i = 0; i < NMAX; ++i) {
            for (j = 0; j < Q; ++j) {
                sum = sum + a[i] + b[i];
            }
        }
        end_time = omp_get_wtime();
        total_time = total_time + (end_time - start_time);
    }
    sum = sum / (ITERS_NUMBER * Q);
    printf("\nREDUCTION SUM : %.0f", sum);

    return total_time / ITERS_NUMBER;
}

double atomic_time(int* a, int* b) {
    double sum = 0, start_time, end_time, total_time = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = omp_get_wtime(); 

#pragma omp parallel for shared(a, b) private(i, j)
        for (i = 0; i < NMAX; ++i) {
            for (j = 0; j < Q; ++j) {
#pragma omp atomic 
                sum += a[i] + b[i];
            }
        }
        end_time = omp_get_wtime();
        total_time = total_time + (end_time - start_time);
    }
    sum = sum / (ITERS_NUMBER * Q);
    printf("\nATOMIC SUM    : %.0f", sum);

    return total_time / ITERS_NUMBER;
}

double critical_time(int* a, int* b) {
    double sum = 0, start_time, end_time, total_time = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = omp_get_wtime();

#pragma omp parallel for shared(a, b) private(i, j)
        for (i = 0; i < NMAX; ++i) {
            for (j = 0; j < Q; ++j) {
#pragma omp critical
                {
                    sum = sum + a[i] + b[i];
                }
            }
        }
        end_time = omp_get_wtime();
        total_time = total_time + (end_time - start_time);
    }

    sum = sum / (ITERS_NUMBER * Q);
    printf("\nCRITICAL SUM  : %.0f\n", sum);

    return total_time / ITERS_NUMBER;
}

double init_time() {
    double sum = 0, start_time, end_time, total_time = 0;
    int cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; cntr++) {
        start_time = omp_get_wtime();
#pragma omp parallel
        {}
        end_time = omp_get_wtime();
        total_time = total_time + (end_time - start_time);
    }

    return total_time / ITERS_NUMBER;
}

int main() {
    omp_set_num_threads(5);

    printf(">>> OpenMP\n");
    printf("1. DATA TYPE      : int\n");
    printf("2. VECTORS SIZE   : %d\n", NMAX);
    printf("3. VECTORS NUMBER : %d\n", 2);
    printf("4. THREADS NUMBER : %d\n", omp_get_max_threads());
    printf("5. PARAMETER Q    : %d\n\n", Q);


    int a[NMAX];
    int b[NMAX];

    int i;
    for (i = 0; i < NMAX; ++i) {
        a[i] = 1;
        b[i] = 2;
    }

    double itime = init_time();
    double stime = sequental_time(a, b);
    double rtime = reduction_time(a, b);
    double atime = atomic_time(a, b);
    double ctime = critical_time(a, b);

    printf("\n>>> TIME OF WORK\n");
    printf("SEQUENTAL : %f\n", stime);
    printf("INIT      : %f\n", itime);
    printf("REDUCTION : %f\n", rtime);
    printf("ATOMIC    : %f\n", atime);
    printf("CRITICAL  : %f\n\n", ctime);

    printf("\n>>> ACCELERATION W/I INIT\n");
    printf("REDUCTION      : %f\n", stime / rtime);
    printf("ATOMIC         : %f\n", stime / atime);
    printf("CRITICAL       : %f\n\n", stime / ctime);

    printf("\n>>> ACCELERATION W/O INIT\n");
    printf("REDUCTION      : %f\n", stime / (rtime - itime));
    printf("ATOMIC         : %f\n", stime / (atime - itime));
    printf("CRITICAL       : %f\n\n", stime / (ctime - itime));

    return 0;
}
