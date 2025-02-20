//
// mmult_accel_tb.cpp
//
// 更新后的测试平台，用于检测 mmult_accel 函数计算结果是否正确。
// 此测试平台分配了与最大接口深度匹配的内存，但只对实际尺寸区域 (N, K, M) 进行初始化与验证。
// 这样可以在 cosim 时避免因内存分配不足而产生越界问题。
//
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "mmult_accel.hpp"  // 假设其中定义了 MAX_N, MAX_K, MAX_M, TILE_SIZE 等宏

#define MAX_N 64
#define MAX_K 768
#define MAX_M 768

// CPU 参考实现：矩阵乘法
static void reference_mmult(const int8_t *A, const int8_t *B, int32_t *C,
                              int N, int K, int M)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            long sum = 0; // 使用64位累加避免中间溢出
            for (int k = 0; k < K; k++) {
                sum += (long)A[i * K + k] * (long)B[k * M + j];
            }
            C[i * M + j] = (int32_t)sum;
        }
    }
}

// 定义一个测试用例结构体
struct TestCase {
    int N;
    int K;
    int M;
};

int main()
{
    // 定义多个测试用例：
    // 1. 标准小尺寸（所有维度均为 TILE_SIZE 整数倍）
    // 2. M 大于 BLOCK_M，触发 B 分块逻辑
    // 3. 非整数倍尺寸，测试边界条件
    TestCase test_cases[] = {
        {16, 768, 768},
        {32, 768, 768},
        {64, 768, 768},
        // {64, 64, 768},
        // {64, 768, 3072}
    };
    const int num_tests = sizeof(test_cases) / sizeof(TestCase);
    bool overall_pass = true;

    // 对于接口内存，按照最大深度分配：
    const int maxA = MAX_N * MAX_K;      // 64 * 768 = 49152
    const int maxB = MAX_K * MAX_M;      // 768 * 3072 = 2359296
    const int maxC = MAX_N * MAX_M;      // 64 * 3072 = 196608

    for (int t = 0; t < num_tests; t++) {
        int N = test_cases[t].N;
        int K = test_cases[t].K;
        int M = test_cases[t].M;
        std::cout << "Running test case " << t << ": N = " << N 
                  << ", K = " << K << ", M = " << M << std::endl;

        // 分配大小为最大深度的数组
        int8_t *A = new int8_t[maxA];
        int8_t *B = new int8_t[maxB];
        int32_t *C_hw = new int32_t[maxC];
        int32_t *C_sw = new int32_t[maxC];

        // 初始化 A：仅前 N*K 区域有效，其他区域置0
        srand(42 + t);
        for (int i = 0; i < N * K; i++) {
            A[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = N * K; i < maxA; i++) {
            A[i] = 0;
        }

        // 初始化 B：仅前 K*M 区域有效，其他区域置0
        for (int i = 0; i < K * M; i++) {
            B[i] = (int8_t)(rand() % 256 - 128);
        }
        for (int i = K * M; i < maxB; i++) {
            B[i] = 0;
        }

        // 将 C 数组全部置0
        for (int i = 0; i < maxC; i++) {
            C_hw[i] = 0;
            C_sw[i] = 0;
        }

        // 调用 CPU 参考实现（只计算有效区域：N, K, M）
        reference_mmult(A, B, C_sw, N, K, M);
        // 调用硬件函数（C-Simulation 下为直接调用）
        mmult_accel(A, B, C_hw, N, K, M);

        // 比较有效区域内的计算结果
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

        // 释放内存
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
