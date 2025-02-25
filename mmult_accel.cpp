#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// Define maximum matrix dimensions (consider BRAM resource limitations)
#define MAX_N 64
#define MAX_K 768
#define MAX_M 768

// Define the width of B blocks
#define BLOCK_M 128

// Matrix tile size fixed to 8 (as per your current settings)
const int TILE_SIZE = 8;

//-----------------------------------------------------------------
// Stage 1: Load a TILE_SIZE x TILE_SIZE block from A_bram
static void load_A_tile_k(const int8_t A_bram[MAX_N][MAX_K],
                            hls::stream<int8_t> &sA,
                            int i0, int k0, int N, int K) {
    for (int ii = 0; ii < TILE_SIZE; ii++) {
        for (int kk = 0; kk < TILE_SIZE; kk++) {
#pragma HLS PIPELINE II=1
            int global_i = i0 + ii;
            int global_k = k0 + kk;
            int8_t a_val = (global_i < N && global_k < K) ? A_bram[global_i][global_k] : 0;
            sA.write(a_val);
        }
    }
}

//-----------------------------------------------------------------
// Stage 2: Load a TILE_SIZE x TILE_SIZE block from B_local
static void load_B_tile_k(const int8_t B_local[MAX_K][BLOCK_M],
                            hls::stream<int8_t> &sB,
                            int j0, int k0, int current_block_M, int K) {
    for (int kk = 0; kk < TILE_SIZE; kk++) {
        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS PIPELINE II=1
            int global_k = k0 + kk;
            int global_j = j0 + jj;
            int8_t b_val = (global_k < K && global_j < current_block_M) ? B_local[global_k][global_j] : 0;
            sB.write(b_val);
        }
    }
}

//-----------------------------------------------------------------
// Stage 3: Read the full tile from the streams and perform compute.
static void compute_tile_k(hls::stream<int8_t> &sA,
                           hls::stream<int8_t> &sB,
                           int32_t localC[TILE_SIZE][TILE_SIZE]) {
    // Local buffers for the tiles
    int8_t A_tile[TILE_SIZE][TILE_SIZE];
    int8_t B_tile[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=A_tile dim=2 complete
#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 complete

    // Read TILE_SIZE*TILE_SIZE elements from sA into A_tile
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int k = 0; k < TILE_SIZE; k++) {
#pragma HLS PIPELINE II=1
            A_tile[i][k] = sA.read();
        }
    }
    // Read TILE_SIZE*TILE_SIZE elements from sB into B_tile
    for (int k = 0; k < TILE_SIZE; k++) {
        for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS PIPELINE II=1
            B_tile[k][j] = sB.read();
        }
    }

    // Perform multiply-accumulate for this k–tile
    for (int kk = 0; kk < TILE_SIZE; kk++) {
        for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS UNROLL
                localC[i][j] += (int32_t)A_tile[i][kk] * (int32_t)B_tile[kk][j];
            }
        }
    }
}

//-----------------------------------------------------------------
// Helper function to process one k–tile for a given C–tile.
// This function overlaps loading from A_bram and B_local with compute.
static void tile_k_compute(const int8_t A_bram[MAX_N][MAX_K],
                             const int8_t B_local[MAX_K][BLOCK_M],
                             int32_t localC[TILE_SIZE][TILE_SIZE],
                             int i0, int j0, int k0,
                             int N, int K, int current_block_M) {
    hls::stream<int8_t> streamA("streamA");
    hls::stream<int8_t> streamB("streamB");
#pragma HLS STREAM variable=streamA depth=64
#pragma HLS STREAM variable=streamB depth=64

#pragma HLS DATAFLOW
    load_A_tile_k(A_bram, streamA, i0, k0, N, K);
    load_B_tile_k(B_local, streamB, j0, k0, current_block_M, K);
    compute_tile_k(streamA, streamB, localC);
}

//-----------------------------------------------------------------
// Top-level matrix multiply kernel with DATAFLOW for the inner k–tile.
extern "C"
{
    void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C,
                     int N, int K, int M)
    {
        // HLS interface configuration
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
            // 3. Compute the corresponding part of C for the current B block
            // *********************************************
            tile_i:
            for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
                tile_j:
                for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {

                    // Define and initialize the local C tile.
                    int32_t localC[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

                    init_c:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
#pragma HLS UNROLL
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
#pragma HLS UNROLL
                            localC[ii][jj] = 0;
                        }
                    }

                    // Process the k–tiles for the current C–tile.
                    k_loop:
                    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                        tile_k_compute(A_bram, B_local, localC, i0, j0, k0, N, K, current_block_M);
                    }

                    // Write the computed C–tile back to external memory.
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
