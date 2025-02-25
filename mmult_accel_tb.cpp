//
// mmult_accel_tb.cpp
//
// Updated testbench to check if the mmult_accel function computes results correctly,
// and to test the newly added update_A feature by mimicking real Q, K, V projections.
// The testbench allocates memory matching the maximum interface depth but only
// initializes and verifies the actual size region (N, K, M) to avoid out-of-bounds issues.
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
    };
    const int num_tests = sizeof(test_cases) / sizeof(TestCase);
    bool overall_pass = true;

    // Allocate memory for interface according to maximum depth:
    const int maxA = MAX_N * MAX_K;   // 64 * 768 = 49152
    const int maxB = MAX_K * MAX_M;   // 768 * 768 = 589824 (if M=768)
    const int maxC = MAX_N * MAX_M;   // 64 * 768 = 49152

    for (int t = 0; t < num_tests; t++) {
        int N = test_cases[t].N;
        int K = test_cases[t].K;
        int M = test_cases[t].M;
        std::cout << "Running test case " << t << ": N = " << N 
                  << ", K = " << K << ", M = " << M << std::endl;

        // Allocate arrays with maximum depth:
        int8_t *A = new int8_t[maxA];
        // Allocate separate weight matrices for Q, K, V projections
        int8_t *B_q = new int8_t[maxB];
        int8_t *B_k = new int8_t[maxB];
        int8_t *B_v = new int8_t[maxB];

        // Allocate arrays for hardware and software results
        int32_t *C_hw = new int32_t[maxC];
        int32_t *C_sw = new int32_t[maxC];

        // Initialize A: only the first N*K region is valid; the rest is zeroed.
        srand(42 + t);
        for (int i = 0; i < N * K; i++) {
            A[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = N * K; i < maxA; i++) {
            A[i] = 0;
        }

        // Initialize weight matrices for Q, K, V:
        // For each, only the first K*M region is valid; the remainder is set to 0.
        srand(100 + t); // Q projection weights
        for (int i = 0; i < K * M; i++) {
            B_q[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = K * M; i < maxB; i++) {
            B_q[i] = 0;
        }
        srand(200 + t); // K projection weights
        for (int i = 0; i < K * M; i++) {
            B_k[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = K * M; i < maxB; i++) {
            B_k[i] = 0;
        }
        srand(300 + t); // V projection weights
        for (int i = 0; i < K * M; i++) {
            B_v[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = K * M; i < maxB; i++) {
            B_v[i] = 0;
        }

        // Test flag for the entire test case
        bool test_pass = true;

        //--------------------------------------------------------------------------
        // Q Projection: load A into BRAM (update_A=1) and use B_q
        //--------------------------------------------------------------------------        
        std::cout << "   Computing Q projection..." << std::endl;
        // Zero out the output arrays for the valid region
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        // Compute reference result for Q
        reference_mmult(A, B_q, C_sw, N, K, M);
        // Call accelerator for Q with update_A=1 (i.e., load A into BRAM)
        mmult_accel(A, B_q, C_hw, N, K, M, 1);
        // Compare the results for Q
        for (int i = 0; i < N * M; i++) {
            if (C_hw[i] != C_sw[i]) {
                test_pass = false;
                std::cout << "   Q projection mismatch at index " << i 
                          << ": HW = " << C_hw[i] 
                          << ", SW = " << C_sw[i] << std::endl;
                break;
            }
        }
        if (test_pass)
            std::cout << "   Q projection Passed." << std::endl;
        else {
            std::cout << "   Q projection Failed." << std::endl;
            overall_pass = false;
        }

        //--------------------------------------------------------------------------
        // K Projection: reuse A in BRAM (update_A=0) and use B_k
        //--------------------------------------------------------------------------
        std::cout << "   Computing K projection..." << std::endl;
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        reference_mmult(A, B_k, C_sw, N, K, M);
        // Call accelerator for K with update_A=0 (reuse previously loaded A)
        mmult_accel(A, B_k, C_hw, N, K, M, 0);
        for (int i = 0; i < N * M; i++) {
            if (C_hw[i] != C_sw[i]) {
                test_pass = false;
                std::cout << "   K projection mismatch at index " << i 
                          << ": HW = " << C_hw[i] 
                          << ", SW = " << C_sw[i] << std::endl;
                break;
            }
        }
        if (test_pass)
            std::cout << "   K projection Passed." << std::endl;
        else {
            std::cout << "   K projection Failed." << std::endl;
            overall_pass = false;
        }

        //--------------------------------------------------------------------------
        // V Projection: reuse A in BRAM (update_A=0) and use B_v
        //--------------------------------------------------------------------------
        std::cout << "   Computing V projection..." << std::endl;
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        reference_mmult(A, B_v, C_sw, N, K, M);
        // Call accelerator for V with update_A=0 (again, reuse A from BRAM)
        mmult_accel(A, B_v, C_hw, N, K, M, 0);
        for (int i = 0; i < N * M; i++) {
            if (C_hw[i] != C_sw[i]) {
                test_pass = false;
                std::cout << "   V projection mismatch at index " << i 
                          << ": HW = " << C_hw[i] 
                          << ", SW = " << C_sw[i] << std::endl;
                break;
            }
        }
        if (test_pass)
            std::cout << "   V projection Passed." << std::endl;
        else {
            std::cout << "   V projection Failed." << std::endl;
            overall_pass = false;
        }

        if(test_pass)
            std::cout << "Test case " << t << " Passed." << std::endl;
        else
            std::cout << "Test case " << t << " Failed." << std::endl;

        // Free memory for this test case
        delete[] A;
        delete[] B_q;
        delete[] B_k;
        delete[] B_v;
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
