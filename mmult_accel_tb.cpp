//
// mmult_accel_tb.cpp
//
// A simple C++ test bench that runs in Vitis/Vivado HLS C Simulation.
// It tests the mmult_accel function with randomly generated int8 matrices
// to verify correctness against a software reference implementation.
//
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mmult_accel.hpp"

// A simple CPU reference function for matrix multiplication
static void reference_mmult(const int8_t *A, const int8_t *B, int32_t *C,
                            int N, int K, int M)
{
    // CPU reference implementation for verification
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            long sum = 0; // accumulate in 64-bit to avoid intermediate overflow
            for (int k = 0; k < K; k++)
            {
                sum += (long)A[i * K + k] * (long)B[k * M + j];
            }
            C[i * M + j] = (int32_t)sum; // cast back to int32
        }
    }
}

int main()
{
    // Test dimensions for a small example
    const int N = 16;
    const int K = 16;
    const int M = 16;

    // Allocate host arrays (in an HLS test bench, normal new/malloc is fine)
    int8_t *A = new int8_t[N * K];
    int8_t *B = new int8_t[K * M];
    int32_t *C_hw = new int32_t[N * M];
    int32_t *C_sw = new int32_t[N * M];

    // Initialize input data
    srand(42);
    for (int i = 0; i < N * K; i++)
    {
        A[i] = (int8_t)(rand() % 256 - 128);
    }
    for (int i = 0; i < K * M; i++)
    {
        B[i] = (int8_t)(rand() % 256 - 128);
    }

    // Zero the output arrays
    for (int i = 0; i < N * M; i++)
    {
        C_hw[i] = 0;
        C_sw[i] = 0;
    }

    // Call reference software mmult
    reference_mmult(A, B, C_sw, N, K, M);

    // Call hardware function (in C-Simulation, this is just a direct call).
    mmult_accel(A, B, C_hw, N, K, M);

    // Compare results
    bool pass = true;
    for (int i = 0; i < N * M; i++)
    {
        if (C_hw[i] != C_sw[i])
        {
            pass = false;
            std::cout << "Mismatch at index " << i
                      << ": HW=" << C_hw[i]
                      << ", SW=" << C_sw[i] << std::endl;
            break;
        }
    }

    if (pass)
    {
        std::cout << "Test Passed. HW matches SW reference.\n";
    }
    else
    {
        std::cout << "Test Failed.\n";
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C_hw;
    delete[] C_sw;

    return pass ? 0 : 1;
}
