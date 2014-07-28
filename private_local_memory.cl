
__kernel //__attribute__ ((reqd_work_group_size(2048, 2048, 1)))
void private_local_memory(__global int* a, __global int* b, __global int* output, __local int* B_local)
{
    int r = get_global_id(0);
    int c, k, index, running;
    int rank = get_global_size(0);
    int iloc = get_local_id(0); 
    int nloc = get_local_size(0);

    int A_private[4096];

    for(index = 0; index < rank; index++){
        A_private[index] = a[r*rank + index];
    }

    for (c=0; c < rank; c++) {
        for (k = iloc; k < rank; k += nloc){
            B_local[k] = b[k*rank+c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        running  = 0;
        for(index = 0; index <  rank; index++)
            running +=  A_private[index] * B_local[index];
        output[r*rank + c] = running;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return;
}
