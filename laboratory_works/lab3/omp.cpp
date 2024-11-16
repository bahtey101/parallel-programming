#include <omp.h>
#include <stdio.h>
#include <cstdlib>
#include <numeric>

#define NMAX         1000
#define ADDITION_DIM 33461
#define CHUNK        100
#define ITERS_NUMBER 20

#define HEADER "| %-10s| %-10s| %-9s| %-14s| %-22s|\n"
#define FORMAT "| %-10s| %-10.f| %-9f| %-14f| %-22f|\n"

double sequental_time(int* a, int* b, int* s, const int &Q) {
   double start_time, total_time = 0;
   int i, j, cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       start_time = omp_get_wtime();

       for (i = 0; i < NMAX; ++i) {
           for (j = 0; j < Q; ++j) {
               s[i] = a[i] + b[i];
           }
       }
       total_time = total_time + (omp_get_wtime() - start_time);
   }
   return total_time / ITERS_NUMBER;
}

double init_time() {
   double start_time, total_time = 0;
   int cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; cntr++) {
       start_time = omp_get_wtime();

#pragma omp parallel
       {}
       total_time = total_time + (omp_get_wtime() - start_time);
   }
   return total_time / ITERS_NUMBER;
}

double static_time(int* a, int* b, int* s, const int& Q) {
   double start_time, total_time = 0;
   int i, j, cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       start_time = omp_get_wtime();

#pragma omp parallel for shared(a, b, s) private(i, j) schedule(static, CHUNK)
       for (i = 0; i < NMAX; ++i) {
           for (j = 0; j < Q; ++j) {
               s[i] = a[i] + b[i];
           }
       }
       total_time = total_time + (omp_get_wtime() - start_time);
   }
   return total_time / ITERS_NUMBER;
}

double dynamic_time(int* a, int* b, int* s, const int& Q) {
   double start_time, total_time = 0;
   int i, j, cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       start_time = omp_get_wtime();

#pragma omp parallel for shared(a, b, s) private(i, j) schedule(dynamic, CHUNK)
       for (i = 0; i < NMAX; ++i) {
           for (j = 0; j < Q; ++j) {
               s[i] = a[i] + b[i];
           }
       }
       total_time = total_time + (omp_get_wtime() - start_time);
   }
   return total_time / ITERS_NUMBER;
}

double guided_time(int* a, int* b, int* s, const int& Q) {
   double start_time, total_time = 0;
   int i, j, cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       start_time = omp_get_wtime();

#pragma omp parallel for shared(a, b, s) private(i, j) schedule(guided, CHUNK)
       for (i = 0; i < NMAX; ++i) {
           for (j = 0; j < Q; ++j) {
               s[i] = a[i] + b[i];
           }
       }
       total_time = total_time + (omp_get_wtime() - start_time);
   }
   return total_time / ITERS_NUMBER;
}

int main(int argc, char* argv[]) {
   int num_threads = 5; // [5, 10, 15]
   int Q           = 3; // 27

   if (argc > 2) {
       num_threads = atoi(argv[1]);
       Q =           atoi(argv[2]);
   }

   omp_set_num_threads(num_threads);

   printf("---------- OpenMP ----------\n");
   printf("  DATA TYPE        : %s\n", "int");
   printf("  VECTORS SIZE     : %d\n", NMAX);
   printf("  ADDITION TO DIM  : %d\n", ADDITION_DIM);
   printf("  VECTORS NUMBER   : %d\n", 2);
   printf("  THREADS NUMBER   : %d\n", num_threads);
   printf("  PARAMETER Q      : %d\n", Q);
   printf("----------------------------\n\n");

   int a[NMAX];
   int b[NMAX];
   int s[NMAX];

   for (int i = 0; i < NMAX; ++i) {
       a[i] = 1;
       b[i] = 2;
   }

   double itime  = init_time();

   double sqtime = sequental_time(a, b, s, Q);
   double sqsum  = std::accumulate(std::begin(s), std::end(s), 0);

   double stime  = static_time   (a, b, s, Q);
   double ssum   = std::accumulate(std::begin(s), std::end(s), 0);

   double dtime  = dynamic_time  (a, b, s, Q);
   double dsum   = std::accumulate(std::begin(s), std::end(s), 0);

   double gtime  = guided_time   (a, b, s, Q);
   double gsum   = std::accumulate(std::begin(s), std::end(s), 0);


   printf(HEADER, "OPTION", "SUM", "TIME", "ACCELERATION", "ACCELERATION W/O INIT");
   printf("|-----------|-----------|----------|---------------|-----------------------|\n");
   printf(FORMAT, "sequental", sqsum, sqtime, 1., 1.);
   printf(FORMAT, "static",    ssum,  stime, sqtime / stime, sqtime / (stime - itime));
   printf(FORMAT, "dynamic",   dsum,  dtime, sqtime / dtime, sqtime / (dtime - itime));
   printf(FORMAT, "guided",    gsum,  gtime, sqtime / gtime, sqtime / (gtime - itime));
   printf("\n> init      | Time : %f\n", itime);

   return 0;
}