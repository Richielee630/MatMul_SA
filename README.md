# Matrix Multiplication Hardware Accelerator

This project implements a hardware-accelerated matrix multiplication using High-Level Synthesis (HLS). The design is optimized for FPGA implementation and utilizes on-chip BRAM for efficient data storage and processing.

## Project Structure

- **mmult_accel.cpp**: Contains the main hardware function `mmult_accel` which performs matrix multiplication.
- **mmult_accel_tb.cpp**: Contains the testbench for verifying the functionality of the `mmult_accel` function.

## Files

### mmult_accel.cpp

This file defines the hardware-accelerated matrix multiplication function. Key features include:

- **HLS Interface Configuration**: Configures the AXI interfaces for the input and output matrices.
- **BRAM Storage**: Uses on-chip BRAM to store the input matrix A and blocks of matrix B.
- **Tiling and Blocking**: Divides the matrices into smaller tiles and blocks for efficient processing.
- **Pipeline and Unroll**: Uses HLS pragmas to pipeline and unroll loops for better performance.

### mmult_accel_tb.cpp

This file contains the testbench for the hardware function. Key features include:

- **Reference Implementation**: A CPU-based reference implementation of matrix multiplication for verification.
- **Test Cases**: Defines multiple test cases with different matrix sizes to verify the hardware function.
- **Memory Allocation**: Allocates memory for the matrices and initializes them with random values.
- **Result Comparison**: Compares the results from the hardware function with the reference implementation.

## How to Run

1. **Setup HLS Environment**: Ensure you have the necessary HLS tools installed and configured.
2. **Compile the Hardware Function**: Use the HLS tool to compile `mmult_accel.cpp` and generate the hardware description.
3. **Run the Testbench**: Compile and run `mmult_accel_tb.cpp` to verify the functionality of the hardware function.

## Example

To compile and run the testbench, use the following commands:

```sh
# Compile the hardware function
vitis_hls -f mmult_accel.cpp

# Compile the testbench
g++ -o mmult_accel_tb mmult_accel_tb.cpp -I/path/to/hls/include

# Run the testbench
./mmult_accel_tb
