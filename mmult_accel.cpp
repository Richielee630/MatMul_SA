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
    // The 'update_A' flag indicates when to reload matrix A from DDR.
    void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C,
                     int N, int K, int M, int update_A)
    {
        //********************************************************************
        // AXI Memory Interface Pragma Declarations:
        // - m_axi interfaces: Connects arrays A, B, and C to external DDR memory.
        // - 'depth' specifies the maximum number of elements to be transferred.
        //********************************************************************
        #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = MAX_N * MAX_K
        #pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = MAX_K * MAX_M
        #pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = MAX_N * MAX_M

        //********************************************************************
        // AXI-Lite Interface Pragma Declarations:
        // - s_axilite interfaces: For passing control arguments (A, B, C, N, K, M, update_A)
        // - 'return' port: Allows the host to know when the accelerator function has completed.
        //********************************************************************
        #pragma HLS INTERFACE s_axilite port = A bundle = control
        #pragma HLS INTERFACE s_axilite port = B bundle = control
        #pragma HLS INTERFACE s_axilite port = C bundle = control
        #pragma HLS INTERFACE s_axilite port = N bundle = control
        #pragma HLS INTERFACE s_axilite port = K bundle = control
        #pragma HLS INTERFACE s_axilite port = M bundle = control
        #pragma HLS INTERFACE s_axilite port = update_A bundle = control
        #pragma HLS INTERFACE s_axilite port = return bundle = control

        //********************************************************************
        // Persistent On-Chip Storage for Matrix A:
        // - 'static' ensures that A_bram retains its content across function calls.
        // - BIND_STORAGE binds A_bram to dual-port BRAM for efficient on-chip storage.
        //********************************************************************
        static int8_t A_bram[MAX_N][MAX_K];
        #pragma HLS BIND_STORAGE variable=A_bram type=ram_2p impl=bram

        // Reload matrix A from DDR only if update_A flag is set
        if(update_A)
        {
            copy_A:
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    // Pipeline this loop to achieve an initiation interval (II) of 1.
                    #pragma HLS PIPELINE II=1
                    A_bram[i][k] = A[i * K + k];
                }
            }
        }

        //********************************************************************
        // Process Matrix B (weights) and Matrix C (output) in column blocks:
        //********************************************************************
        outer_j_block:
        for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
            int current_block_M = ((j_block + BLOCK_M) <= M) ? BLOCK_M : (M - j_block);

            // Local buffer for a block of matrix B (dimensions: [K][BLOCK_M])
            // Bind B_bram to dual-port BRAM for fast on-chip access.
            int8_t B_bram[MAX_K][BLOCK_M];
            #pragma HLS BIND_STORAGE variable=B_bram type=ram_2p impl=bram

            copy_B_block:
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < current_block_M; j++) {
                    // Pipeline this loop to maintain high throughput.
                    #pragma HLS PIPELINE II=1
                    B_bram[k][j] = B[k * M + (j_block + j)];
                }
            }

            //****************************************************************
            // Compute the corresponding part of C for the current B block.
            // This computation is tiled by both rows (i0) and columns (j0).
            //****************************************************************
            tile_i:
            for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
                tile_j:
                for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {

                    // Local buffer for a tile of output matrix C.
                    // Fully partition localC for maximum parallelism.
                    int32_t localC[TILE_SIZE][TILE_SIZE];
                    #pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

                    // Initialize the local C tile to 0.
                    // Fully unroll the initialization loops because localC is fully partitioned (residing in registers),
                    // which enables all 256 elements to be set to zero in parallel with minimal resource overhead.
                    init_c:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
                        #pragma HLS UNROLL
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
                            #pragma HLS UNROLL
                            localC[ii][jj] = 0;
                        }
                    }

                    // Local buffers for tiles from matrix A and B.
                    int8_t localA[TILE_SIZE][TILE_SIZE];
                    int8_t localB[TILE_SIZE][TILE_SIZE];
                    // Fully partition localA along its first dimension (rows) to enable 
                    // parallel access across different rows during multiplication,
                    #pragma HLS ARRAY_PARTITION variable=localA dim=1 complete

                    // Fully partition localB along its second dimension (columns) to allow 
                    // parallel access across different columns.
                    #pragma HLS ARRAY_PARTITION variable=localB dim=2 complete

                    //****************************************************************
                    // Traverse the K dimension in chunks of TILE_SIZE.
                    // For each tile, load data from A_bram and B_bram to local buffers,
                    // then compute the multiplication.
                    //****************************************************************
                    k_loop:
                    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                        // Load a tile from A_bram into localA.
                        // Pipeline this loop with an initiation interval (II) of 1 to ensure high throughput.
                        // Pipelining here minimizes resource usage compared to unrolling, 
                        // which is important for handling external memory accesses.
                        loadA:
                        for (int ii = 0; ii < TILE_SIZE; ii++) {
                            for (int kk = 0; kk < TILE_SIZE; kk++) {
                                // Pipeline this loop for high throughput.
                                #pragma HLS PIPELINE II=1
                                int global_i = i0 + ii;
                                int global_k = k0 + kk;
                                if (global_i < N && global_k < K)
                                    localA[ii][kk] = A_bram[global_i][global_k];
                                else
                                    localA[ii][kk] = 0;
                            }
                        }

                        // Load a tile from B_bram into localB.
                        // Similarly, pipeline this loop with II=1 to balance high throughput with resource usage,
                        // avoiding the resource-intensive replication that full unrolling 
                        // would require for external memory accesses.
                        loadB:
                        for (int kk = 0; kk < TILE_SIZE; kk++) {
                            for (int jj = 0; jj < TILE_SIZE; jj++) {
                                #pragma HLS PIPELINE II=1
                                int global_k = k0 + kk;
                                int global_j = j0 + jj;
                                if (global_k < K && global_j < current_block_M)
                                    localB[kk][jj] = B_bram[global_k][global_j];
                                else
                                    localB[kk][jj] = 0;
                            }
                        }

                        // Compute the matrix multiplication for the current tile:
                        // localC = localC + localA * localB.
                        compute:
                        for (int kk = 0; kk < TILE_SIZE; kk++) {
                            #pragma HLS PIPELINE II=1
                            for (int ii = 0; ii < TILE_SIZE; ii++) {
                                #pragma HLS UNROLL
                                // Compute a_val once per (kk, ii) since localA[ii][kk] is invariant for the entire jj loop.
                                // Declaring a_val here ensures that each unrolled iteration gets its own register,
                                // which maximizes parallelism and avoids redundant computations.
                                int32_t a_val = (int32_t)localA[ii][kk];
                                for (int jj = 0; jj < TILE_SIZE; jj++) {
                                    #pragma HLS UNROLL 
                                    // Compute b_val within the jj loop because localB[kk][jj] changes with jj.
                                    // Declaring b_val here also guarantees that each iteration of the unrolled loop 
                                    // has its own independent value, enabling the HLS tool to optimize the pipeline effectively.
                                    int32_t b_val = (int32_t)localB[kk][jj];
                                    localC[ii][jj] += a_val * b_val;
                                }
                            }
                        }
                    } // End of k_loop

                    // Write the computed tile of matrix C back to DDR.
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
