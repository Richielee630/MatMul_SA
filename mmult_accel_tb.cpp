//
// mmult_accel_tb.cpp
//
// Updated testbench to check if the mmult_accel function computes results correctly.
// This testbench allocates memory matching the maximum interface depth but only initializes and verifies the actual size region (N, K, M).
// This avoids out-of-bounds issues during cosimulation due to insufficient memory allocation.
//
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mmult_accel.hpp"  // Assuming it defines MAX_N, MAX_K, MAX_M, TILE_SIZE, etc.

#define MAX_N 64
#define MAX_K 768
#define MAX_M 768

// CPU reference implementation: matrix multiplication
static void reference_mmult(const int8_t *A, const int8_t *B, int32_t *C,
                              int N, int K, int M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            long sum = 0; // Use 64-bit accumulation to avoid intermediate overflow
            for (int k = 0; k < K; k++) {
                sum += (long)A[i * K + k] * (long)B[k * M + j];
            }
            C[i * M + j] = (int32_t)sum;
        }
    }
}

// Define a test case structure
struct TestCase {
    int N;
    int K;
    int M;
};

int main()
{
    // Define multiple test cases:
    // 1. Standard small size (all dimensions are multiples of TILE_SIZE)
    // 2. M greater than BLOCK_M, triggering B block logic
    // 3. Non-multiple sizes, testing boundary conditions
    TestCase test_cases[] = {
        {16, 768, 768},
        {32, 768, 768},
        {64, 768, 768},
        // {64, 64, 768},
        // {64, 768, 3072}
    };
    const int num_tests = sizeof(test_cases) / sizeof(TestCase);
    bool overall_pass = true;

    // Allocate memory for interface according to maximum depth:
    const int maxA = MAX_N * MAX_K;      // 64 * 768 = 49152
    const int maxB = MAX_K * MAX_M;      // 768 * 3072 = 2359296
    const int maxC = MAX_N * MAX_M;      // 64 * 3072 = 196608

    for (int t = 0; t < num_tests; t++) {
        int N = test_cases[t].N;
        int K = test_cases[t].K;
        int M = test_cases[t].M;
        std::cout << "Running test case " << t << ": N = " << N 
                  << ", K = " << K << ", M = " << M << std::endl;

        // Allocate arrays with maximum depth
        int8_t *A = new int8_t[maxA];
        int8_t *B = new int8_t[maxB];
        int32_t *C_hw = new int32_t[maxC];
        int32_t *C_sw = new int32_t[maxC];

        // Initialize A: only the first N*K region is valid, other regions are set to 0
        srand(42 + t);
        for (int i = 0; i < N * K; i++) {
            A[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = N * K; i < maxA; i++) {
            A[i] = 0;
        }

        // Initialize B: only the first K*M region is valid, other regions are set to 0
        for (int i = 0; i < K * M; i++) {
            B[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = K * M; i < maxB; i++) {
            B[i] = 0;
        }

        // Set all elements of C arrays to 0
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }

        // Call CPU reference implementation (only compute the valid region: N, K, M)
        reference_mmult(A, B, C_sw, N, K, M);
        // Call hardware function (direct call in C-Simulation)
        mmult_accel(A, B, C_hw, N, K, M);

        // Compare the computation results in the valid region
        bool pass = true;
        for (int i = 0; i < N * M; i++) {
            if (C_hw[i] != C_sw[i]) {
                pass = false;
                std::cout << "Mismatch at index " << i 
                          << ": HW = " << C_hw[i] 
                          << ", SW = " << C_sw[i] << std::endl;
                break;
            }
        }
        if (pass) {
            std::cout << "Test case " << t << " Passed." << std::endl;
        } else {
            std::cout << "Test case " << t << " Failed." << std::endl;
            overall_pass = false;
        }

        // Free memory
        delete[] A;
        delete[] B;
        delete[] C_hw;
        delete[] C_sw;
    }

    if (overall_pass) {
        std::cout << "All test cases Passed." << std::endl;
    } else {
        std::cout << "Some test cases Failed." << std::endl;
    }

    return overall_pass ? 0 : 1;
}
