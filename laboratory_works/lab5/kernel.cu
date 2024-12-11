#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define ITERATIONS 20
#define NSIZE 3

#define IDXC(i, j, N) ((i) * (N) + (j))
#define IDXF(i, j, N) ((i) + (N) * (j))

void gpu_blas_mmul(const double *A, const double *B, double *C, const int N)
{
	int lda = N, ldb = N, ldc = N;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, alpha, A, lda, B, ldb, beta, C, ldc);
	cublasDestroy(handle);
}

void cpu_blas_mmul(const double *A, const double *B, double *C, const int N)
{
	const double alpha = 1;
	const double beta = 0;

	int i, j, k;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			C[i * N + j] = 0;
			for (k = 0; k < N; k++)
			{
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = alpha * C[i * N + j] + beta;
		}
	}
}

void gpu_fill_rand(double *A, int m, int n)
{
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
	curandGenerateUniformDouble(prng, A, m * n);
	cudaDeviceSynchronize();
	curandDestroyGenerator(prng);
}

void cpu_fill_rand(double *A, int m, int n)
{
	int i;
	for (i = 0; i < m * n; i++)
	{
		A[i] = (double)rand() / RAND_MAX;
	}
}

void print_row(double *A, int i, int N, bool column_major)
{
	printf("%f %f ... %f %f\n",
		   A[column_major ? IDXF(i, 0, N) : IDXC(i, 0, N)],
		   A[column_major ? IDXF(i, 1, N) : IDXC(i, 1, N)],
		   A[column_major ? IDXF(i, N - 2, N) : IDXC(i, N - 2, N)],
		   A[column_major ? IDXF(i, N - 1, N) : IDXC(i, N - 1, N)]);
}

void print_matrix(double *A, int N, bool column_major = false)
{
	print_row(A, 0, N, column_major);
	print_row(A, 1, N, column_major);
	printf("...\n");
	print_row(A, N - 2, N, column_major);
	print_row(A, N - 1, N, column_major);
	printf("\n");
}

int main()
{
	int sizes[NSIZE] = {512, 1024, 2048};

	printf("----------- CUDA -----------\n");
	printf("  DATA TYPE        : double\n");
	printf("  MATRIX SIZE      : [512, 1024, 2048]\n");
	printf("  TRANSPOSE        : [T, T]\n");
	printf("----------------------------\n\n");

	int i, k;
	float time;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double t_mh[NSIZE] = {0, 0, 0};
	double t_md[NSIZE] = {0, 0, 0};
	double t_s[NSIZE] = {0, 0, 0};
	double t_trhost[NSIZE] = {0, 0, 0};
	double t_trdev[NSIZE] = {0, 0, 0};
	double t_cu[NSIZE] = {0, 0, 0};

	double *A, *B, *C;
	double *A_dev, *B_dev, *C_dev;

	for (i = 0; i < NSIZE; i++)
	{

		int N = sizes[i];
		printf("Size of the matrices: %d\n", N);

		int nbytes = N * N * sizeof(double);
		for (k = 0; k < ITERATIONS; k++)
		{
			A = (double *)malloc(nbytes);
			B = (double *)malloc(nbytes);
			C = (double *)malloc(nbytes);

			cudaMalloc(&A_dev, nbytes);
			cudaMalloc(&B_dev, nbytes);
			cudaMalloc(&C_dev, nbytes);

			cudaEventRecord(start);
			cpu_fill_rand(A, N, N);
			cpu_fill_rand(B, N, N);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_mh[i] += time;

			cudaEventRecord(start);
			cudaMemcpy(A_dev, A, nbytes, cudaMemcpyHostToDevice);
			cudaMemcpy(B_dev, B, nbytes, cudaMemcpyHostToDevice);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_trdev[i] += time;

			cudaEventRecord(start);
			gpu_fill_rand(A_dev, N, N);
			gpu_fill_rand(B_dev, N, N);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_md[i] += time;

			cudaMemcpy(A, A_dev, nbytes, cudaMemcpyDeviceToHost);
			cudaMemcpy(B, B_dev, nbytes, cudaMemcpyDeviceToHost);

			if (k == ITERATIONS - 1)
			{
				printf("A =\n");
				print_matrix(A, N);
				printf("B =\n");
				print_matrix(B, N);
			}

			cudaEventRecord(start);
			cpu_blas_mmul(A, B, C, N);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_s[i] += time;

			if (k == ITERATIONS - 1)
			{
				printf("C (CPU) =\n");
				print_matrix(C, N, false);
			}

			cudaEventRecord(start);
			gpu_blas_mmul(A_dev, B_dev, C_dev, N);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_cu[i] += time;

			cudaEventRecord(start);
			cudaMemcpy(C, C_dev, nbytes, cudaMemcpyDeviceToHost);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			t_trhost[i] += time;

			if (k == ITERATIONS - 1)
			{
				printf("C (GPU) =\n");
				print_matrix(C, N, true);
			}

			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			cudaFree(A_dev);
			cudaFree(B_dev);
			cudaFree(C_dev);

			free(A);
			free(B);
			free(C);
		}
	}

	printf("%-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
		   "size", "t_Mh", "t_Md", "a_Mgen", "t_s", "t_cu", "a_cu", "t_trdev", "t_trhost", "a_Mhcu", "a_Mdcu");

	for (int i = 0; i < NSIZE; i++)
	{
		printf("%-10d %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n",
			   sizes[i],
			   t_mh[i] / ITERATIONS,
			   t_md[i] / ITERATIONS,
			   t_mh[i] / t_md[i],
			   t_s[i] / ITERATIONS,
			   t_cu[i] / ITERATIONS,
			   t_s[i] / t_cu[i],
			   t_trdev[i] / ITERATIONS,
			   t_trhost[i] / ITERATIONS,
			   (t_s[i] + t_mh[i]) / (t_cu[i] + t_mh[i] + t_trdev[i] + t_trhost[i]),
			   (t_s[i] + t_mh[i]) / (t_cu[i] + t_md[i] + t_trhost[i]));
	}
	return 0;
}