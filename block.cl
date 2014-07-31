/*  Matrix Multiplication  a*b = output
    Optimisation using tiling    
    Tiles of size: (BLOCK_SIZE x BLOCK_SIZE)
    Each work group computes one block of C at a time
    Each work group stores the sub-matrices of A and B in local memory
    Each work item computes an element of C = dot product of a row and a column      */

#define BLOCK_SIZE 16
#define AS(i,j) as[j + i*BLOCK_SIZE]
#define BS(i,j) bs[j + i*BLOCK_SIZE]

__kernel __attribute__ ((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void block(__global double* a, __global double* b, __global double* output, __local double* as, __local double* bs)
{
    // Index of the tile/work-group
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Index of the thread within the tile/Index of the work item
    int thread_x = get_local_id(0);
    int thread_y = get_local_id(1);

    int rank = get_global_size(0);
    double running = 0.0f;

   // Starting index for a and b matrices i.e. first sub matrices
    int a_index = rank * BLOCK_SIZE * block_y;
    int b_index = BLOCK_SIZE * block_x;

    // Step sizes for a and b, cycle through
    int a_step = BLOCK_SIZE;
    int b_step = BLOCK_SIZE * rank;

    int end = a_index + (rank - 1);
    int r, c, n;
    
    // Cycle through all sub-matrices of a and b and compute dot products
    for(r = a_index, c = b_index; r < end; r += a_step, c += b_step)
    {
        // Load sub-matrices from global memory to local memory
        AS(thread_y, thread_x) = a[r + rank*thread_y + thread_x];
        BS(thread_y, thread_x) = b[c + rank*thread_y + thread_x];

    // Wait for all work items to load data
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll

        // Dot product
        for(n = 0; n < BLOCK_SIZE; ++n)
            running += AS(thread_y, n)*BS(n, thread_x);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write sub-matrix to device memory
    output[get_global_id(1)*get_global_size(0) + get_global_id(0)] = running;
}
