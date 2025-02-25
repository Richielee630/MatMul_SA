//
// mmult_accel.hpp
//
// Header file defining constants, includes, and function prototypes
// for the 16x16 INT8 systolic array accelerator.
//
#ifndef MMULT_ACCEL_HPP
#define MMULT_ACCEL_HPP

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// Define the systolic array tile size (16x16)
// static const int TILE_SIZE = 16;

// Top-level function prototype for matrix multiplication accelerator.
// This matches the function we defined in mmult_accel.cpp
extern "C"
{
    void mmult_accel(
        const int8_t *A, // pointer to matrix A in DDR
        const int8_t *B, // pointer to matrix B in DDR
        int32_t *C,      // pointer to output matrix C in DDR
        int N,           // rows of A, rows of C
        int K,           // cols of A, rows of B
        int M,            // cols of B, cols of C
        int update_A   // flag to update matrix A from DDR
    );
}

#endif // MMULT_ACCEL_HPP
