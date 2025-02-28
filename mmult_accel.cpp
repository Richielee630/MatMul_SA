#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

/**
 * @brief Matrix multiplication accelerator parameters
 * 
 * These parameters define the maximum dimensions supported by the accelerator
 * and are constrained by available FPGA resources (particularly BRAM).
 */
#define MAX_N 64    // Maximum number of rows in matrix A and output C
#define MAX_K 768   // Maximum shared dimension between matrices A and B
#define MAX_M 768   // Maximum number of columns in matrix B and output C

/** 
 * @brief Memory blocking parameter for column-wise processing
 * 
 * For efficient memory usage, matrix B is processed in blocks of BLOCK_M columns.
 * This reduces on-chip memory requirements while maintaining high throughput.
 */
#define BLOCK_M 256

/**
 * @brief Computation tile size for enhanced parallelism
 * 
 * The tile size determines the dimensions of local buffers used during computation.
 * Larger values increase parallelism but require more FPGA resources.
 */
const int TILE_SIZE = 32;

/**
 * @brief Data type definitions for input and output matrices
 * 
 * Using int8_t for inputs enables efficient storage and computation.
 * int32_t for outputs provides sufficient dynamic range for accumulated results.
 */
#define DTYPE_IN int8_t
#define DTYPE_OUT int32_t

extern "C"
{
    /**
     * @brief Matrix multiplication accelerator with persistent on-chip storage
     * 
     * This function implements an optimized matrix multiplication (C = A×B) where:
     * - Matrix A can be persistent in on-chip memory between calls
     * - Processing is tiled and blocked for optimal FPGA resource utilization
     * - Computation is highly parallelized within each tile
     * 
     * @param A Input matrix A [N×K] (row-major)
     * @param B Input matrix B [K×M] (row-major)
     * @param C Output matrix C [N×M] (row-major)
     * @param N Number of rows in matrix A and output C
     * @param K Shared dimension between matrices A and B
     * @param M Number of columns in matrix B and output C
     * @param update_A Flag to control reloading of matrix A (1=reload, 0=reuse)
     */
    void mmult_accel(const DTYPE_IN *A, const DTYPE_IN *B, DTYPE_OUT *C,
                     int N, int K, int M, int update_A)
    {
        //********************************************************************
        // AXI Memory Interface Configuration:
        // Connects matrices A, B, C to external memory through separate channels
        // with specified depth limits for memory transfers
        //********************************************************************
        #pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmemA depth = MAX_N * MAX_K
        #pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmemB depth = MAX_K * MAX_M
        #pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmemC depth = MAX_N * MAX_M

        //********************************************************************
        // AXI-Lite Control Interface Configuration:
        // Provides host control over function parameters and execution state
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
        // Static allocation ensures data persistence across function calls
        // Dual-port BRAM implementation enables parallel access
        //********************************************************************
        static DTYPE_IN A_bram[MAX_N][MAX_K];
        #pragma HLS BIND_STORAGE variable=A_bram type=ram_2p impl=bram

        // Conditionally reload matrix A from external memory to on-chip BRAM
        // This optimization allows reusing A across multiple calculations
        if(update_A)
        {
            copy_A:
            for (int i = 0; i < N; i++) {
                for (int k = 0; k < K; k++) {
                    #pragma HLS PIPELINE II=1
                    A_bram[i][k] = A[i * K + k];
                }
            }
        }

        //********************************************************************
        // Block-based Processing of Matrix B:
        // Process matrix B in column blocks to optimize memory usage
        // Each block is loaded into on-chip BRAM for efficient access
        //********************************************************************
        outer_j_block:
        for (int j_block = 0; j_block < M; j_block += BLOCK_M) {
            int current_block_M = ((j_block + BLOCK_M) <= M) ? BLOCK_M : (M - j_block);

            // Allocate on-chip storage for current block of matrix B
            DTYPE_IN B_bram[MAX_K][BLOCK_M];
            #pragma HLS BIND_STORAGE variable=B_bram type=ram_2p impl=bram

            // Load current block of matrix B into on-chip BRAM
            copy_B_block:
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < current_block_M; j++) {
                    #pragma HLS PIPELINE II=1
                    B_bram[k][j] = B[k * M + (j_block + j)];
                }
            }

            //****************************************************************
            // Tiled Matrix Multiplication:
            // Divide computation into tiles for both row and column dimensions
            // This approach optimizes data reuse and computation parallelism
            //****************************************************************
            tile_i:
            for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
                tile_j:
                for (int j0 = 0; j0 < current_block_M; j0 += TILE_SIZE) {

                    // Allocate and partition register array for output tile
                    // Complete partitioning transforms the array into individual registers
                    // for maximum parallel access
                    DTYPE_OUT localC[TILE_SIZE][TILE_SIZE];
                    #pragma HLS ARRAY_PARTITION variable=localC dim=0 complete

                    // Initialize output tile to zero
                    // Full unrolling enables parallel initialization of all elements
                    init_c:
                    for (int ii = 0; ii < TILE_SIZE; ii++) {
                        #pragma HLS UNROLL
                        for (int jj = 0; jj < TILE_SIZE; jj++) {
                            #pragma HLS UNROLL
                            localC[ii][jj] = 0;
                        }
                    }

                    // Allocate and partition register arrays for input tiles
                    // Complete partitioning enables parallel access to all elements
                    DTYPE_IN localA[TILE_SIZE][TILE_SIZE];
                    DTYPE_IN localB[TILE_SIZE][TILE_SIZE];
                    #pragma HLS ARRAY_PARTITION variable=localA dim=0 complete
                    #pragma HLS ARRAY_PARTITION variable=localB dim=0 complete

                    //****************************************************************
                    // K-dimension Tiling:
                    // Process shared dimension K in tiles to maximize data reuse
                    // For each K-tile, load data into local buffers then compute
                    //****************************************************************
                    k_loop:
                    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                        // Load tile from matrix A into local buffer
                        // Pipelined implementation balances throughput and resource usage
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

                        // Load tile from matrix B into local buffer
                        // Pipelined for efficient memory access
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

                        // Compute matrix multiplication for current tile
                        // Highly optimized with pipelining and unrolling for maximum throughput
                        compute:
                        for (int kk = 0; kk < TILE_SIZE; kk++) {
                            #pragma HLS PIPELINE II=1
                            for (int ii = 0; ii < TILE_SIZE; ii++) {
                                #pragma HLS UNROLL
                                // Pre-load and cast A value to output type once per iteration
                                DTYPE_OUT a_val = (DTYPE_OUT)localA[ii][kk];
                                for (int jj = 0; jj < TILE_SIZE; jj++) {
                                    #pragma HLS UNROLL 
                                    // Cast B value to output type and accumulate product
                                    DTYPE_OUT b_val = (DTYPE_OUT)localB[kk][jj];
                                    localC[ii][jj] += a_val * b_val;
                                }
                            }
                        }
                    } // End of k_loop

                    // Write computed tile back to external memory
                    // Boundary checking ensures valid data only is written
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
