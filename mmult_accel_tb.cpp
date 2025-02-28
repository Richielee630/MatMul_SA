/**
 * @file mmult_accel_tb.cpp
 * @brief Testbench for the optimized matrix multiplication accelerator
 *
 * This testbench validates the functionality and performance of the mmult_accel
 * accelerator, with particular focus on testing the persistent on-chip storage
 * feature (controlled by update_A). The test mimics a typical transformer architecture
 * pattern where the same input is projected using three different weight matrices
 * (Q, K, V projections).
 *
 * Safety features:
 * - Allocates memory according to maximum interface depths
 * - Performs boundary checks to avoid out-of-bounds access
 * - Supports both functional validation and performance measurement
 * - FAST_COSIM mode for rapid co-simulation with smaller datasets
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>  // Added for timing measurements
#include "mmult_accel.hpp"  // Assuming it defines MAX_N, MAX_K, MAX_M, TILE_SIZE, etc.

/**
 * @brief Maximum matrix dimensions supported by the accelerator
 * 
 * These must match the values defined in the accelerator implementation.
 */
#define MAX_N 64
#define MAX_K 768
#define MAX_M 768

/**
 * @brief HLS co-simulation cycle-accurate timing support
 *
 * When running in HLS co-simulation mode, these functions provide
 * cycle-accurate performance measurements.
 */
#ifdef HLS_COSIM
#include "ap_int.h"
extern "C" ap_uint<64> clock_start();
extern "C" ap_uint<64> clock_end();
#endif

/**
 * @brief Software reference implementation of matrix multiplication
 *
 * This function provides a baseline CPU implementation for validating
 * accelerator results. It uses 64-bit intermediate accumulation to
 * prevent overflow during computation.
 *
 * @param A Input matrix A [N×K] in row-major order
 * @param B Input matrix B [K×M] in row-major order
 * @param C Output matrix C [N×M] in row-major order
 * @param N Number of rows in matrix A and output C
 * @param K Shared dimension between matrices A and B
 * @param M Number of columns in matrix B and output C
 */
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

/**
 * @brief Test case structure defining matrix dimensions to test
 */
struct TestCase {
    int N;  ///< Number of rows in matrix A and output C
    int K;  ///< Shared dimension between matrices A and B
    int M;  ///< Number of columns in matrix B and output C
};

/**
 * @brief Main testbench function
 *
 * Executes a series of test cases to validate the accelerator's:
 * 1. Functional correctness - comparing outputs with reference implementation
 * 2. Performance characteristics - measuring GFLOPs across different matrix sizes
 * 3. Persistent storage feature - testing A matrix reuse across multiple operations
 *
 * @return 0 if all tests pass, 1 if any test fails
 */
int main()
{
    // Define test cases with different matrix dimensions
    // In FAST_COSIM mode, use only a minimal test case for faster simulation
#ifdef FAST_COSIM
    #pragma message("FAST_COSIM mode: running minimal test case for quick cosim")
    TestCase test_cases[] = {
        {8, 64, 64}
    };
#else
    TestCase test_cases[] = {
        {16, 768, 768},
        {32, 768, 768},
        {64, 768, 768},
    };
#endif
    const int num_tests = sizeof(test_cases) / sizeof(TestCase);
    bool overall_pass = true;
    
    // Performance tracking variables
    double total_gflops = 0.0;
    int total_measurements = 0;
    double best_gflops = 0.0;
    TestCase best_case = {0, 0, 0};

    // Allocate memory according to maximum interface depths
    // This ensures we don't exceed HLS interface depth specifications
    const int maxA = MAX_N * MAX_K;   // 64 * 768 = 49,152 elements
    const int maxB = MAX_K * MAX_M;   // 768 * 768 = 589,824 elements
    const int maxC = MAX_N * MAX_M;   // 64 * 768 = 49,152 elements

    // Process each test case
    for (int t = 0; t < num_tests; t++) {
        int N = test_cases[t].N;
        int K = test_cases[t].K;
        int M = test_cases[t].M;
        std::cout << "Running test case " << t << ": N = " << N 
                  << ", K = " << K << ", M = " << M << std::endl;

        // Allocate arrays with maximum depth to satisfy HLS interface requirements
        int8_t *A = new int8_t[maxA];
        // Allocate separate weight matrices for Q, K, V projections
        int8_t *B_q = new int8_t[maxB];
        int8_t *B_k = new int8_t[maxB];
        int8_t *B_v = new int8_t[maxB];

        // Allocate arrays for hardware and software results
        int32_t *C_hw = new int32_t[maxC];
        int32_t *C_sw = new int32_t[maxC];

        // Initialize matrix data with deterministic pseudorandom values
        // Only initialize the actual used portions (N*K, K*M) and zero the rest
        srand(42 + t);
        for (int i = 0; i < N * K; i++) {
            A[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = N * K; i < maxA; i++) {
            A[i] = 0;
        }

        // Initialize weight matrices for Q, K, V projections
        // Using different seeds for diversity in test patterns
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

        // Test flag for the current test case
        bool test_pass = true;
        
        // Calculate theoretical operation count for performance metrics
        // Matrix multiplication requires 2*N*M*K operations (N*M*K multiplies and N*M*K adds)
        double ops_per_mmult = 2.0 * N * M * K;

        //--------------------------------------------------------------------------
        // Q Projection: load A into BRAM (update_A=1) and use B_q
        // This first projection requires loading matrix A into on-chip memory
        //--------------------------------------------------------------------------        
        std::cout << "   Computing Q projection..." << std::endl;
        // Initialize output arrays
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        
        // Compute reference implementation result
        reference_mmult(A, B_q, C_sw, N, K, M);
        
        // Measure accelerator performance using appropriate timing mechanism
        #ifdef HLS_COSIM
        ap_uint<64> cycle_start = clock_start();
        #endif
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Call accelerator with update_A=1 to load matrix A into BRAM
        mmult_accel(A, B_q, C_hw, N, K, M, 1);
        
        // Calculate execution time and performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        #ifdef HLS_COSIM
        ap_uint<64> cycle_end = clock_end();
        ap_uint<64> cycles_taken = cycle_end - cycle_start;
        double clock_period_ns = 10.0; // Assuming 100MHz clock (10ns)
        double execution_time_s = (double)(cycles_taken.to_uint64()) * clock_period_ns * 1.0e-9;
        double gflops_q = ops_per_mmult / (execution_time_s * 1.0e9);
        std::cout << "   Q projection cycles: " << cycles_taken.to_uint64() 
                  << ", GFLOPs: " << gflops_q << std::endl;
        #else
        std::chrono::duration<double> elapsed = end_time - start_time;
        double gflops_q = ops_per_mmult / (elapsed.count() * 1.0e9);
        std::cout << "   Q projection time: " << elapsed.count() << " s"
                  << ", GFLOPs: " << gflops_q << std::endl;
        #endif
        
        // Track performance statistics
        total_gflops += gflops_q;
        total_measurements++;
        
        if (gflops_q > best_gflops) {
            best_gflops = gflops_q;
            best_case = test_cases[t];
        }
        
        // Validate results by comparing accelerator output with reference
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
        // This second projection reuses the already loaded matrix A
        //--------------------------------------------------------------------------
        std::cout << "   Computing K projection..." << std::endl;
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        
        // Compute reference implementation result
        reference_mmult(A, B_k, C_sw, N, K, M);
        
        // Measure accelerator performance
        #ifdef HLS_COSIM
        cycle_start = clock_start();
        #endif
        start_time = std::chrono::high_resolution_clock::now();
        
        // Call accelerator with update_A=0 to reuse previously loaded matrix A
        mmult_accel(A, B_k, C_hw, N, K, M, 0);
        
        // Calculate execution time and performance metrics
        end_time = std::chrono::high_resolution_clock::now();
        #ifdef HLS_COSIM
        cycle_end = clock_end();
        cycles_taken = cycle_end - cycle_start;
        double execution_time_s_k = (double)(cycles_taken.to_uint64()) * clock_period_ns * 1.0e-9;
        double gflops_k = ops_per_mmult / (execution_time_s_k * 1.0e9);
        std::cout << "   K projection cycles: " << cycles_taken.to_uint64() 
                  << ", GFLOPs: " << gflops_k << std::endl;
        #else
        elapsed = end_time - start_time;
        double gflops_k = ops_per_mmult / (elapsed.count() * 1.0e9);
        std::cout << "   K projection time: " << elapsed.count() << " s"
                  << ", GFLOPs: " << gflops_k << std::endl;
        #endif
        
        // Track performance statistics
        total_gflops += gflops_k;
        total_measurements++;
        
        if (gflops_k > best_gflops) {
            best_gflops = gflops_k;
            best_case = test_cases[t];
        }
        
        // Validate results
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
        // This third projection again reuses the already loaded matrix A
        //--------------------------------------------------------------------------
        std::cout << "   Computing V projection..." << std::endl;
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }
        
        // Compute reference implementation result
        reference_mmult(A, B_v, C_sw, N, K, M);
        
        // Measure accelerator performance
        #ifdef HLS_COSIM
        cycle_start = clock_start();
        #endif
        start_time = std::chrono::high_resolution_clock::now();
        
        // Call accelerator with update_A=0 to reuse previously loaded matrix A
        mmult_accel(A, B_v, C_hw, N, K, M, 0);
        
        // Calculate execution time and performance metrics
        end_time = std::chrono::high_resolution_clock::now();
        #ifdef HLS_COSIM
        cycle_end = clock_end();
        cycles_taken = cycle_end - cycle_start;
        double execution_time_s_v = (double)(cycles_taken.to_uint64()) * clock_period_ns * 1.0e-9;
        double gflops_v = ops_per_mmult / (execution_time_s_v * 1.0e9);
        std::cout << "   V projection cycles: " << cycles_taken.to_uint64() 
                  << ", GFLOPs: " << gflops_v << std::endl;
        #else
        elapsed = end_time - start_time;
        double gflops_v = ops_per_mmult / (elapsed.count() * 1.0e9);
        std::cout << "   V projection time: " << elapsed.count() << " s"
                  << ", GFLOPs: " << gflops_v << std::endl;
        #endif
        
        // Track performance statistics
        total_gflops += gflops_v;
        total_measurements++;
        
        if (gflops_v > best_gflops) {
            best_gflops = gflops_v;
            best_case = test_cases[t];
        }
        
        // Validate results
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

        // Report test case result summary
        if(test_pass)
            std::cout << "Test case " << t << " Passed." << std::endl;
        else
            std::cout << "Test case " << t << " Failed." << std::endl;

        // Release allocated memory
        delete[] A;
        delete[] B_q;
        delete[] B_k;
        delete[] B_v;
        delete[] C_hw;
        delete[] C_sw;
    }

    // Report overall test results
    if (overall_pass) {
        std::cout << "All test cases Passed." << std::endl;
    } else {
        std::cout << "Some test cases Failed." << std::endl;
    }
    
    // Report consolidated performance statistics
    double avg_gflops = total_gflops / total_measurements;
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "  Average GFLOPs: " << avg_gflops << std::endl;
    std::cout << "  Best GFLOPs: " << best_gflops << " (N=" << best_case.N 
              << ", K=" << best_case.K << ", M=" << best_case.M << ")" << std::endl;

    return overall_pass ? 0 : 1;
}
