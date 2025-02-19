#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// Matrix dimensions are given at runtime via AXI4-Lite,
// but we fix the systolic array tile size as 16:
const int TILE_SIZE = 16;

// Top-level HLS function for matrix multiplication accelerator
extern "C"
{
    void mmult_accel(const int8_t *A, const int8_t *B, int32_t *C,
                     int N, int K, int M)
    {
// HLS interface pragmas for AXI4 master and AXI4-Lite
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem0 depth = 256
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = N bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = M bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // Local buffers for a 16x16 tile of A, B, and C.
        // Double-buffering for A and B (ping-pong) and single buffer for C accumulation.
        int8_t localA_ping[TILE_SIZE][TILE_SIZE];
        int8_t localA_pong[TILE_SIZE][TILE_SIZE];
        int8_t localB_ping[TILE_SIZE][TILE_SIZE];
        int8_t localB_pong[TILE_SIZE][TILE_SIZE];
        int32_t localC[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = localA_ping dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = localA_pong dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = localB_ping dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = localB_pong dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete
    // (localC dim=0 complete -> fully partition all dimensions)

    // Iterate over output tiles C[i0:i0+15][j0:j0+15]
    tile_i:
        for (int i0 = 0; i0 < N; i0 += TILE_SIZE)
        {
        tile_j:
            for (int j0 = 0; j0 < M; j0 += TILE_SIZE)
            {
            // Initialize localC tile to 0
            init_c:
                for (int ii = 0; ii < TILE_SIZE; ++ii)
                {
#pragma HLS UNROLL
                    for (int jj = 0; jj < TILE_SIZE; ++jj)
                    {
#pragma HLS UNROLL
                        localC[ii][jj] = 0;
                    }
                }

                // Ping-pong buffer management for K dimension tiles
                bool usePing = true;
                int k0 = 0;
                // Preload the first A and B 16x16 tile into ping buffers
                {
                loadA_init:
                    for (int ii = 0; ii < TILE_SIZE; ++ii)
                    {
                        for (int kk = 0; kk < TILE_SIZE; ++kk)
                        {
#pragma HLS PIPELINE II = 1
                            // Global index in matrix A = (i0+ii, k0+kk)
                            if (i0 + ii < N && k0 + kk < K)
                                localA_ping[ii][kk] = A[(i0 + ii) * K + (k0 + kk)];
                            else
                                localA_ping[ii][kk] = 0; // zero-padding for out-of-bound
                        }
                    }
                loadB_init:
                    for (int kk = 0; kk < TILE_SIZE; ++kk)
                    {
                        for (int jj = 0; jj < TILE_SIZE; ++jj)
                        {
#pragma HLS PIPELINE II = 1
                            // Global index in matrix B = (k0+kk, j0+jj)
                            if (k0 + kk < K && j0 + jj < M)
                                localB_ping[kk][jj] = B[(k0 + kk) * M + (j0 + jj)];
                            else
                                localB_ping[kk][jj] = 0;
                        }
                    }
                }
                // Advance k0 to first loaded tile size
                k0 += TILE_SIZE;
                usePing = false; // next, we'll use pong for loading

                // Loop over K in chunks of 16 (tiles) and compute
                while (k0 < K)
                {
                    // Load next tile of A and B into the buffer (pong if usePing is false, ping if true)
                    if (usePing)
                    {
                    // Load into ping buffers (while compute will use pong)
                    loadA_ping:
                        for (int ii = 0; ii < TILE_SIZE; ++ii)
                        {
                            for (int kk = 0; kk < TILE_SIZE; ++kk)
                            {
#pragma HLS PIPELINE II = 1
                                int global_i = i0 + ii;
                                int global_k = k0 + kk;
                                if (global_i < N && global_k < K)
                                    localA_ping[ii][kk] = A[global_i * K + global_k];
                                else
                                    localA_ping[ii][kk] = 0;
                            }
                        }
                    loadB_ping:
                        for (int kk = 0; kk < TILE_SIZE; ++kk)
                        {
                            for (int jj = 0; jj < TILE_SIZE; ++jj)
                            {
#pragma HLS PIPELINE II = 1
                                int global_k = k0 + kk;
                                int global_j = j0 + jj;
                                if (global_k < K && global_j < M)
                                    localB_ping[kk][jj] = B[global_k * M + global_j];
                                else
                                    localB_ping[kk][jj] = 0;
                            }
                        }
                    }
                    else
                    {
                    // Load into pong buffers (while compute will use ping)
                    loadA_pong:
                        for (int ii = 0; ii < TILE_SIZE; ++ii)
                        {
                            for (int kk = 0; kk < TILE_SIZE; ++kk)
                            {
#pragma HLS PIPELINE II = 1
                                int global_i = i0 + ii;
                                int global_k = k0 + kk;
                                if (global_i < N && global_k < K)
                                    localA_pong[ii][kk] = A[global_i * K + global_k];
                                else
                                    localA_pong[ii][kk] = 0;
                            }
                        }
                    loadB_pong:
                        for (int kk = 0; kk < TILE_SIZE; ++kk)
                        {
                            for (int jj = 0; jj < TILE_SIZE; ++jj)
                            {
#pragma HLS PIPELINE II = 1
                                int global_k = k0 + kk;
                                int global_j = j0 + jj;
                                if (global_k < K && global_j < M)
                                    localB_pong[kk][jj] = B[global_k * M + global_j];
                                else
                                    localB_pong[kk][jj] = 0;
                            }
                        }
                    }

                    // Compute multiplication on the previously loaded tile (ping or pong buffer)
                    if (usePing)
                    {
                    // usePing true here means we just loaded ping, so compute on pong
                    compute_pong:
                        for (int kk = 0; kk < TILE_SIZE; ++kk)
                        {
#pragma HLS PIPELINE II = 1
                            for (int ii = 0; ii < TILE_SIZE; ++ii)
                            {
#pragma HLS UNROLL
                                for (int jj = 0; jj < TILE_SIZE; ++jj)
                                {
#pragma HLS UNROLL
                                    // MAC: localC[ii][jj] += localA_pong[ii][kk] * localB_pong[kk][jj];
                                    int32_t a_val = (int32_t)localA_pong[ii][kk];
                                    int32_t b_val = (int32_t)localB_pong[kk][jj];
                                    localC[ii][jj] += a_val * b_val;
                                }
                            }
                        }
                    }
                    else
                    {
                    // usePing false here means we just loaded pong, so compute on ping
                    compute_ping:
                        for (int kk = 0; kk < TILE_SIZE; ++kk)
                        {
#pragma HLS PIPELINE II = 1
                            for (int ii = 0; ii < TILE_SIZE; ++ii)
                            {
#pragma HLS UNROLL
                                for (int jj = 0; jj < TILE_SIZE; ++jj)
                                {
#pragma HLS UNROLL
                                    int32_t a_val = (int32_t)localA_ping[ii][kk];
                                    int32_t b_val = (int32_t)localB_ping[kk][jj];
                                    localC[ii][jj] += a_val * b_val;
                                }
                            }
                        }
                    }

                    // Toggle buffer usage for next iteration and advance k0
                    usePing = !usePing;
                    k0 += TILE_SIZE;
                } // end while for K tiles

                // After the loop, one last tile remains computed but not yet accumulated if k0 == K
                // We need to perform the compute for the last loaded tile (when K is a multiple of 16, the last tile compute happens here).
                if (!usePing)
                {
                // If usePing is false, it means the last loaded tile data is in ping (because we toggled one extra time at loop exit),
                // so compute with ping data:
                compute_last_ping:
                    for (int kk = 0; kk < TILE_SIZE; ++kk)
                    {
#pragma HLS PIPELINE II = 1
                        for (int ii = 0; ii < TILE_SIZE; ++ii)
                        {
#pragma HLS UNROLL
                            for (int jj = 0; jj < TILE_SIZE; ++jj)
                            {
#pragma HLS UNROLL
                                int32_t a_val = (int32_t)localA_ping[ii][kk];
                                int32_t b_val = (int32_t)localB_ping[kk][jj];
                                localC[ii][jj] += a_val * b_val;
                            }
                        }
                    }
                }
                else
                {
                // If usePing is true, the last loaded tile is in pong buffer
                compute_last_pong:
                    for (int kk = 0; kk < TILE_SIZE; ++kk)
                    {
#pragma HLS PIPELINE II = 1
                        for (int ii = 0; ii < TILE_SIZE; ++ii)
                        {
#pragma HLS UNROLL
                            for (int jj = 0; jj < TILE_SIZE; ++jj)
                            {
#pragma HLS UNROLL
                                int32_t a_val = (int32_t)localA_pong[ii][kk];
                                int32_t b_val = (int32_t)localB_pong[kk][jj];
                                localC[ii][jj] += a_val * b_val;
                            }
                        }
                    }
                }

            // Write the 16x16 result tile from localC back to global memory C
            writeC:
                for (int ii = 0; ii < TILE_SIZE; ++ii)
                {
                    for (int jj = 0; jj < TILE_SIZE; ++jj)
                    {
#pragma HLS PIPELINE II = 1
                        if (i0 + ii < N && j0 + jj < M)
                        {
                            C[(i0 + ii) * M + (j0 + jj)] = localC[ii][jj];
                        }
                    }
                }
            } // tile_j
        } // tile_i
    } // mmult_accel
} // extern "C"
