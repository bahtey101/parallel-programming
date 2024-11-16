#include "mpi.h"
#include <stdio.h>

#define NMAX         1000
#define ITERS_NUMBER 20
#define Q            27

double sequental_time(int* a, int* b) {
   double sum = 0, start_time, end_time, total_time = 0;
   int i, j, cntr;

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       start_time = MPI_Wtime();
       for (i = 0; i < NMAX; ++i) {
           for (j = 0; j < Q; ++j) {
               sum = sum + a[i] + b[i];
           }
       }
       end_time = MPI_Wtime();
       total_time = total_time + (end_time - start_time);
   }
   sum = sum / (ITERS_NUMBER * Q);
   return total_time / ITERS_NUMBER;
}

int main(int argc, char* argv[]) {
   double p2p_sum = 0, reduce_sum = 0, proc_sum = 0.0;

   int a[NMAX];
   int b[NMAX];

   int proc_rank, proc_num, i, j, cntr;
   int k, i1, i2;
   MPI_Status status;

   double bcast_time = 0, p2p_time = 0, reduce_time = 0;
   double time_point, stime;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
   MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

   if (proc_rank == 0) {
       printf(">>> MPI\n");
       printf("1. DATA TYPE      : int\n");
       printf("2. VECTORS SIZE   : %d\n", NMAX);
       printf("3. VECTORS NUMBER : %d\n", 2);
       printf("4. PROC NUMBER    : %d\n", proc_num); // [5, 10, 15]
       printf("5. PARAMETER Q    : %d\n", Q);

       int i;
       for (i = 0; i < NMAX; ++i) {
           a[i] = 1;
           b[i] = 2;
       }

       stime = sequental_time(a, b);
   }

   for (cntr = 0; cntr < ITERS_NUMBER; ++cntr) {
       time_point = MPI_Wtime();
       MPI_Bcast(a, NMAX, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(b, NMAX, MPI_INT, 0, MPI_COMM_WORLD);
       bcast_time += (MPI_Wtime() - time_point) / ITERS_NUMBER;

       time_point = MPI_Wtime();
       proc_sum = 0;
       k = NMAX / proc_num;
       i1 = k * proc_rank;
       i2 = k * (proc_rank + 1);

       if (proc_rank == proc_num - 1) i2 = NMAX;
       for (i = i1; i < i2; i++) {
           for (j = 0; j < Q; ++j) {
               proc_sum = proc_sum + a[i] + b[i];
           }
       }

       if (proc_rank == 0)
       {
           p2p_sum = p2p_sum + proc_sum;
           for (i = 1; i < proc_num; i++)
           {
               MPI_Recv(&proc_sum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
               p2p_sum = p2p_sum + proc_sum;
           }
       }
       else {
           MPI_Send(&proc_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
       }
       p2p_time += (MPI_Wtime() - time_point) / ITERS_NUMBER;
       MPI_Barrier(MPI_COMM_WORLD);

       time_point = MPI_Wtime();
       proc_sum = 0;
       k = NMAX / proc_num;
       i1 = k * proc_rank;
       i2 = (proc_rank == proc_num - 1) ? NMAX : k * (proc_rank + 1);

       for (i = i1; i < i2; ++i) {
           for (j = 0; j < Q; ++j) {
               proc_sum = proc_sum + a[i] + b[i];
           }
       }
       MPI_Reduce(&proc_sum, &reduce_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

       reduce_time += (MPI_Wtime() - time_point) / ITERS_NUMBER;
       MPI_Barrier(MPI_COMM_WORLD);
   }
   MPI_Barrier(MPI_COMM_WORLD);

   if (proc_rank == 0) {
       printf("\n>>> POINT TO POINT");
       printf("\nTOTAL SUM              : %8.f", p2p_sum / (Q * ITERS_NUMBER));
       printf("\nTIME OF WORK           : %f", p2p_time);
       printf("\nACCELERATION W/I BCAST : %f", stime / (p2p_time + bcast_time));
       printf("\nACCELERATION W/O BCAST : %f\n", stime / p2p_time);

       printf("\n>>> COLLECTIVE");
       printf("\nTOTAL SUM              : %8.f", reduce_sum / Q);
       printf("\nTIME OF WORK           : %f", reduce_time);
       printf("\nACCELERATION W/I BCAST : %f", stime / (reduce_time + bcast_time));
       printf("\nACCELERATION W/O BCAST : %f\n", stime / reduce_time);

       printf("\nBROADCAST TIME         : %f", bcast_time);
       printf("\nSEQUENTAL TIME         : %f\n", stime);
   }
   MPI_Finalize();
   return 0;
}