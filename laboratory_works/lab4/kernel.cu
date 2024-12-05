#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#define NMAX 6000000
#define ITERATIONS 20

__global__ void addKernel(int* a, int* b, int* s, unsigned int size) {
    int gridSize = blockDim.x * gridDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start; i < size; i += gridSize) {
        s[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    int GRID_DIM  = 128;
    int BLOCK_DIM = 1024;

    if (argc > 2) {
       GRID_DIM  = atoi(argv[1]);
       BLOCK_DIM = atoi(argv[2]);
   }

    int n2i = NMAX * sizeof(int);
    // Выделение памяти на хосте
    int* a = (int*)calloc(NMAX, sizeof(int));
    int* b = (int*)calloc(NMAX, sizeof(int));
    int* s = (int*)calloc(NMAX, sizeof(int));

    // Инициализация массивов
    for (int i = 0; i < NMAX; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    // Выделение памяти на устройстве
    int* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, n2i);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    int* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n2i);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    int* sdev = NULL;
    cuerr = cudaMalloc((void**)&sdev, n2i);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for s: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float seqTime = 0.0f;
    float gpuTime = 0.0f;
    float trTime  = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    for(int cntr = 0; cntr < ITERATIONS; ++cntr) {
        // Замер последовательного алгоритма
        float seqTimetmp = 0.0f;

        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) 
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        for (int i = 0; i < NMAX; ++i) {
            s[i] = a[i] + b[i];
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&seqTimetmp, start, stop);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "1Cannot calculate elapsed time: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }



        // Параллельный алгоритим
        // Замер времени передачи данных на видеокарту
        float to_device_tmp = 0.0f;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) 
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }   

        cuerr = cudaMemcpy(adev, a, n2i, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot copy a array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaMemcpy(bdev, b, n2i, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot copy b array from host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&to_device_tmp, start, stop);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "2Cannot calculate elapsed time: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }


        
        // Замер времени выполнения ядра
        float gpuTimetmp = 0.0f;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) 
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        addKernel <<< GRID_DIM, BLOCK_DIM >>> (adev, bdev, sdev, NMAX);

        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record stop CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "3Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&gpuTimetmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot calculate elapsed time: %s\n", 
                cudaGetErrorString(cuerr));
            return 0;
        }



        // Замер времени передачи данных с видеокарты в ОЗУ
        float to_host_tmp = 0.0f;
        cuerr = cudaEventRecord(start, 0);
        if (cuerr != cudaSuccess) 
        {
            fprintf(stderr, "Cannot record CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaMemcpy(s, sdev, n2i, cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot copy c array from device to host: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventRecord(stop, 0);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot record stop CUDA event: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        cuerr = cudaEventElapsedTime(&to_host_tmp, start, stop);
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "4Cannot calculate elapsed time: %s\n", 
                cudaGetErrorString(cuerr));
            return 0;
        }

        seqTime += seqTimetmp / ITERATIONS;
        gpuTime += gpuTimetmp / ITERATIONS;
        trTime  += (to_device_tmp + to_host_tmp) / ITERATIONS; 
    }

    printf("----------- CUDA -----------\n");
    printf("  DATA TYPE        : %s\n", "int");
    printf("  VECTORS SIZE     : %d\n", NMAX);
    printf("  VECTORS NUMBER   : %d\n", 2);
    printf("  GRID DIM         : %d\n", GRID_DIM);
    printf("  BLOCK DIM        : %d\n", BLOCK_DIM);
    printf("----------------------------\n\n");

    printf(" Sequental time    : %f ms\n",   seqTime);
    printf(" Time to send(tr)  : %f ms\n",   trTime);
    printf(" CUDA time of work : %f ms\n\n", gpuTime);

    printf(" CUDA acceleration w/i send : %f\n",   seqTime / (gpuTime + trTime));
    printf(" CUDA acceleration w/o send : %f\n\n", seqTime / gpuTime);

    printf(" VECTOR <S>\n");
    for (int i = 0; i < 3; ++i) {
        printf(" s[%d] = %d\n", i, s[i]);
    }
    printf("    ...\n");
    for (int i = -3; i < 0; ++i) {
        printf(" s[%d] = %d\n", NMAX + i, s[NMAX + i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(sdev);
    free(a);
    free(b);
    free(s);

    return 0;
}

