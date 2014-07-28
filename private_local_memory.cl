
__kernel //__attribute__ ((reqd_work_group_size(2048, 2048, 1)))
void private_local_memory(__global int* a, __global int* b, __global int* output)
{
    int r = get_global_id(0);
    int c, index, running;
    int rank = get_global_size(0);

    int A_private[4096];

    for(index = 0; index < rank; index++){
        A_private[index] = a[r*rank + index];
    }

    for (c=0; c < rank; c++) {
        running  = 0;
        for(index = 0; index <  rank; index++)
            running +=  A_private[index] * b[index*rank+c];
        output[r*rank + c] = running;
    }

    return;
}
