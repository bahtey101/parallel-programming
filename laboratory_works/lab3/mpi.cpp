#include <mpi.h>
#include <stdio.h>
#include <cstdlib>

#define NMAX         1000
#define ADDITION_DIM 7 //33461
#define ITERS_NUMBER 20

#define HEADER "| %-13s| %-10s| %-9s| %-14s| %-23s|\n"
#define FORMAT "| %-13s| %-10.f| %-9f| %-14f| %-23f|\n"

double sequental_time(int* a, int* b, int* s, const int& Q) {
    double start_time, total_time = 0;
    int i, j, cntr;

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        start_time = MPI_Wtime();

        for (i = 0; i < NMAX; ++i) {
            for (j = 0; j < Q; ++j) {
                s[i] = a[i] + b[i];
            }
        }
        total_time = total_time + (MPI_Wtime() - start_time);
    }
    return total_time / ITERS_NUMBER;
}

int main(int argc, char* argv[]) {
    int proc_num, proc_rank;
    int Q = 1;
    int i, j, cntr, count, N = NMAX + ADDITION_DIM;

    int *a = nullptr, *b = nullptr, *s = nullptr;
    int *a_loc = nullptr, *b_loc = nullptr, *s_loc = nullptr;

    double start_time, stime;
    double sctime = 0, scvtime = 0;
    double multime = 0, nontime = 0;
    double mulsum = 0, nonsum = 0;

    if (argc > 1) {
        Q = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    if (proc_rank == 0) {
        printf("------------ MPI -----------\n");
        printf("  DATA TYPE        : %s\n", "int");
        printf("  VECTORS SIZE     : %d\n", NMAX);
        printf("  ADDITION TO DIM  : %d\n", ADDITION_DIM);
        printf("  VECTORS NUMBER   : %d\n", 2);
        printf("  THREADS NUMBER   : %d\n", proc_num);
        printf("  PARAMETER Q      : %d\n", Q);
        printf("----------------------------\n\n");

        a = (int*)malloc(NMAX * sizeof(int));
        b = (int*)malloc(NMAX * sizeof(int));
        s = (int*)malloc(NMAX * sizeof(int));

        for (i = 0; i < NMAX; ++i) {
            a[i] = 1;
            b[i] = 2;
        }

        stime = sequental_time(a, b, s, Q);
    }
    
    count = NMAX / proc_num;

    a_loc = (int*)malloc(count * sizeof(int));
    b_loc = (int*)malloc(count * sizeof(int));
    s_loc = (int*)malloc(count * sizeof(int));

    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        if (proc_rank == 0) start_time = MPI_Wtime();
        MPI_Scatter(a, count, MPI_INT, a_loc, count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b, count, MPI_INT, b_loc, count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_rank == 0) sctime += MPI_Wtime() - start_time;

        for (i = 0; i < count; ++i) {
            for (j = 0; j < Q; ++j) {
                s_loc[i] = a_loc[i] + b_loc[i];
            }
        }

        MPI_Gather(s_loc, count, MPI_INT, s, count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_rank == 0) multime += MPI_Wtime() - start_time;
    }

    if (proc_rank == 0) {
        for (i = 0; i < NMAX; ++i) {
            mulsum += s[i];
        }

        free(a);
        free(b);
        free(s);
    }
    free(a_loc);
    free(b_loc);
    free(s_loc);

    int* counts = (int*)malloc(proc_num * sizeof(int));
    int* displs = (int*)malloc(proc_num * sizeof(int));
    displs[0] = 0;

    for (i = 0; i < proc_num; ++i) {
        counts[i] = N / proc_num;
        if (i < N % proc_num) {
            counts[i] += 1;
        }

        if (i == 0) continue;

        displs[i] = displs[i - 1] + counts[i - 1];
    }

    if (proc_rank == 0) {
        a = (int*)malloc(N * sizeof(int));
        b = (int*)malloc(N * sizeof(int));
        s = (int*)malloc(N * sizeof(int));

        for (i = 0; i < N; ++i) {
            a[i] = 1;
            b[i] = 2;
        }
    }

    a_loc = (int*)malloc(counts[proc_rank] * sizeof(int));
    b_loc = (int*)malloc(counts[proc_rank] * sizeof(int));
    s_loc = (int*)malloc(counts[proc_rank] * sizeof(int));


    for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
        if (proc_rank == 0) start_time = MPI_Wtime();
        MPI_Scatterv(a, counts, displs, MPI_INT, a_loc, counts[proc_rank], MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(b, counts, displs, MPI_INT, b_loc, counts[proc_rank], MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_rank == 0) scvtime += MPI_Wtime() - start_time;

        for (i = 0; i < counts[proc_rank]; ++i) {
            for (j = 0; j < Q; ++j) {
                s_loc[i] = a_loc[i] + b_loc[i];
            }
        }

        MPI_Gatherv(s_loc, counts[proc_rank], MPI_INT, s, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (proc_rank == 0) nontime += MPI_Wtime() - start_time;
    }
    free(a_loc);
    free(b_loc);
    free(s_loc);
    free(counts);
    free(displs);

    if (proc_rank == 0) {
        for (i = 0; i < N; ++i) {
            nonsum += s[i];
        }

        printf(HEADER, "TYPE", "SUM", "TIME", "ACCELERATION", "ACCELERATION W/O BCAST");
        printf("|--------------|-----------|----------|---------------|------------------------|\n");
        printf(FORMAT, "sequental",  3. * NMAX, stime, 1., 1.);
        printf(FORMAT, "multiple",     mulsum, multime, stime / multime, stime / (multime - sctime));
        printf(FORMAT, "non multiple", nonsum, nontime, stime / nontime, stime / (nontime - scvtime));
        printf("\n> MPI_Scatter  | Time : %f",   sctime);
        printf("\n> MPI_Scatterv | Time : %f\n", scvtime);

        free(a);
        free(b);
        free(s);
    }

    MPI_Finalize();
    return 0;
}