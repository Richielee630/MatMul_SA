#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// Define maximum matrix dimensions (consider BRAM resource limitations)
#define MAX_N 64
#define MAX_K 768
#define MAX_M 768

// Define the width of B blocks
#define BLOCK_M 256

// Matrix tile size fixed to 16
const int TILE_SIZE = 16;

extern "C"
{
    void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C,
                     int N, int K, int M)
    {
        // HLS interface configuration
// #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = 256
// #pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = 256
// #pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = 256
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = MAX_N * MAX_K
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = MAX_K * MAX_M
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = MAX_N * MAX_M
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // *********************************************
        // 1. Copy the entire A matrix to on-chip BRAM
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
        // 2. Process B and C in column blocks
        // *********************************************
        // Outer loop: process B columns in blocks of BLOCK_M
        outer_j_block:
        for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
            // Actual width of the current block (the last block may be smaller than BLOCK_M)
            int current_block_M = ((j_block + BLOCK_M) <= M) ? BLOCK_M : (M - j_block);

            // Define local B block buffer, size [K][BLOCK_M]
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
            // 3. Compute the corresponding part of C for the current B block: C[:, j_block : j_block+current_block_M]
            // *********************************************
            // Divide A rows and current B block columns by TILE_SIZE
            tile_i:
            for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
                tile_j:
                for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {

                    // Define local C tile buffer
                    int32_t localC[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

                    // Initialize C tile to 0
                    init_c:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                            localC[ii][jj] = 0;
                        }
                    }

                    // Define single local A and B tile buffers
                    int8_t localA[TILE_SIZE][TILE_SIZE];
                    int8_t localB[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localA dim=1 complete
#pragma HLS ARRAY_PARTITION variable=localB dim=2 complete

                    // Traverse K dimension, divide by TILE_SIZE, accumulate calculations
                    k_loop:
                    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                        // Load A tile from A_bram to localA
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

                        // Load B tile from B_local to localB
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

                        // Compute: localC += localA * localB
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

                    // Write the computation result back to DDR (note the offset j_block when writing back)
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
