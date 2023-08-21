/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /**
  * Matrix multiplication: C = A * B.
  * Host code.
  *
  * This sample implements matrix multiplication which makes use of shared memory
  * to ensure data reuse, the matrix multiplication is done using tiling approach.
  * It has been written for clarity of exposition to illustrate various CUDA programming
  * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
  * See also:
  * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
  * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
  * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
  */

  // System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define K 2
#define L 3

template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
 
    int aBegin = wA * BLOCK_SIZE * by * K;

    int aEnd = aBegin + wA - 1;

    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx * L;


    int bStep = BLOCK_SIZE * wB;

    float Csub[K * L] = { 0.0f };


    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {

        __shared__ float As[K * BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][L * BLOCK_SIZE];


        for (int i = 0; i < K; ++i) {
            As[ty + BLOCK_SIZE * i][tx] = A[a + wA * (ty + BLOCK_SIZE * i) + tx];
        }
        for (int i = 0; i < L; ++i) {
            Bs[ty][tx + BLOCK_SIZE * i] = B[b + wB * ty + tx + BLOCK_SIZE * i];
        }
       
        __syncthreads();

      

#pragma unroll

       for (int k = 0;k < K;k++) {
            for (int l = 0;l < L;l++) {
                for (int i = 0;i < BLOCK_SIZE;i++) {
                    Csub[k * L + l] += As[ty + BLOCK_SIZE * k][i] * Bs[i][tx + BLOCK_SIZE * l];

                }
            }
        }

        __syncthreads();
    }

    int cstart = aBegin + bBegin + wB * ty + tx;
    for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
            C[cstart + (k * wB * BLOCK_SIZE) + l * BLOCK_SIZE] = Csub[k * L + l];
        }
    }
    

    
}

void ConstantInit(float* data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}
/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char** argv,
    int block_size, const dim3& dimsA,
    const dim3& dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    cudaStream_t stream;

    // Initialize host memory
    const float valB = 0.1f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float* d_A, * d_B, * d_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float* h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // copy host memory to device
    checkCudaErrors(
        cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(
        cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / (threads.x * L), dimsA.x / (threads.y * K));
    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    
    // Performs warmup operation using matrixMul CUDA kernel
    /*if (block_size == 16) {
        MatrixMulCUDA<16>
            << <grid, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
    else {
        MatrixMulCUDA<32>
            << <grid, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
    }*/

    printf("done\n");
    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 1;

    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16>
                << <grid, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else {
            MatrixMulCUDA<32>
                << <grid, threads, 0, stream >> > (d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }
  
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
        static_cast<double>(dimsA.y) *
        static_cast<double>(dimsB.x);
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(
        cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6;  // machine zero
    
    /*printf("\n");
    for (int i = 0;i < dimsA.y;i++) {
        for (int j = 0;j < dimsA.y;j++) {
            printf("%.2f\t ", h_C[j + i * dimsA.y]);
        }
        printf("\n");
    }*/
    
    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf(
        "\nNOTE: The CUDA Samples are not meant for performance "
        "measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct) {
        return EXIT_SUCCESS;
    }
    else {
        return EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char** argv) {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
        checkCmdLineFlag(argc, (const char**)argv, "?")) {
        printf("Usage -device=n (n >= 0 for deviceID)\n");
        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
        printf("  Note: Outer matrix dimensions of A & B matrices" \
            " must be equal.\n");

        exit(EXIT_SUCCESS);
    }

    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char**)argv);

    const int block_size = 32;

    const int matrix_size = block_size * K * L * 5;
 
    dim3 dimsA(matrix_size, matrix_size, 1);
    dim3 dimsB(matrix_size, matrix_size, 1);


    // width of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char**)argv, "wA");
    }

    // height of Matrix A
    if (checkCmdLineFlag(argc, (const char**)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char**)argv, "hA");
    }

    // width of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char**)argv, "wB");
    }

    // height of Matrix B
    if (checkCmdLineFlag(argc, (const char**)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char**)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
            dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
        dimsB.x, dimsB.y);

    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(matrix_result);
}
