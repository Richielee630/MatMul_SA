#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// 定义最大矩阵尺寸（注意BRAM资源限制）
#define MAX_N 64
#define MAX_K 768
#define MAX_M 768


// 定义 B 分块的宽度
#define BLOCK_M 256

// Matrix tile size 固定为16
const int TILE_SIZE = 16;

extern "C"
{
    void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C,
                     int N, int K, int M)
    {
        // HLS 接口配置
// #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = 256
// #pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = 256
// #pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = 256
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = 49152
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = 589824
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = 49152
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // *********************************************
        // 1. 将整个 A 矩阵复制到 on-chip BRAM
        // *********************************************
        int8_t A_bram[MAX_N][MAX_K];
#pragma HLS BIND_STORAGE variable=A_bram type=ram_2p impl=bram

        copy_A:
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
                A_bram[i][k] = A[i * K + k];
            }
        }

        // *********************************************
        // 2. 按列分块处理 B 与 C
        // *********************************************
        // 外层循环：按 BLOCK_M 分块处理 B 的列
        outer_j_block:
        for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
            // 当前块实际宽度（最后一块可能小于 BLOCK_M）
            int current_block_M = ((j_block + BLOCK_M) <= M) ? BLOCK_M : (M - j_block);

            // 定义局部 B 块缓冲区，尺寸为 [K][BLOCK_M]
            int8_t B_local[MAX_K][BLOCK_M];
#pragma HLS BIND_STORAGE variable=B_local type=ram_2p impl=bram

            copy_B_block:
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < current_block_M; j++) {
#pragma HLS PIPELINE II=1
                    B_local[k][j] = B[k * M + (j_block + j)];
                }
            }

            // *********************************************
            // 3. 计算当前 B 块对应的 C 部分：C[:, j_block : j_block+current_block_M]
            // *********************************************
            // 按 TILE_SIZE 划分 A 的行和当前 B 块的列
            tile_i:
            for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
                tile_j:
                for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {

                    // 定义局部 C tile 缓冲区
                    int32_t localC[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

                    // 初始化 C tile 为 0
                    init_c:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                            localC[ii][jj] = 0;
                        }
                    }

                    // 定义单一的局部 A、B tile 缓冲区
                    int8_t localA[TILE_SIZE][TILE_SIZE];
                    int8_t localB[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localA dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localB dim=2 complete

                    // 遍历 K 维度，按 TILE_SIZE 划分，累加计算
                    k_loop:
                    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                        // 加载 A tile 从 A_bram 到 localA
                        loadA:
                        for (int ii = 0; ii < TILE_SIZE; ii++) {
                            for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
                                int global_i = i0 + ii;
                                int global_k = k0 + kk;
                                if (global_i < N && global_k < K)
                                    localA[ii][kk] = A_bram[global_i][global_k];
                                else
                                    localA[ii][kk] = 0;
                            }
                        }

                        // 加载 B tile 从 B_local 到 localB
                        loadB:
                        for (int kk = 0; kk < TILE_SIZE; kk++) {
                            for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS PIPELINE II=1
                                int global_k = k0 + kk;
                                int global_j = j0 + jj;
                                if (global_k < K && global_j < current_block_M)
                                    localB[kk][jj] = B_local[global_k][global_j];
                                else
                                    localB[kk][jj] = 0;
                            }
                        }

                        // 计算： localC += localA * localB
                        compute:
                        for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
                            for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                                for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                                    int32_t a_val = (int32_t)localA[ii][kk];
                                    int32_t b_val = (int32_t)localB[kk][jj];
                                    localC[ii][jj] += a_val * b_val;
                                }
                            }
                        }
                    } // end k_loop

                    // 将计算结果写回 DDR（写回时注意偏移 j_block）
                    writeC:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS PIPELINE II=1
                            int global_i = i0 + ii;
                            int global_j = j0 + jj;
                            if (global_i < N && global_j < current_block_M)
                                C[global_i * M + (j_block + global_j)] = localC[ii][jj];
                        }
                    }
                }
            }
        }
    }
}
