//#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel //__attribute__ ((reqd_work_group_size(2048, 2048, 1)))
void private_memory(__global double* a, __global double* b, __global double* output)
{
    int r = get_global_id(0);
    int c, index;
    double running;
    int rank = get_global_size(0);

    double A_private[4096];

    for(index = 0; index < rank; index++){
        A_private[index] = a[r*rank + index];
    }

    for (c=0; c < rank; c++) {
        running  = 0.0f;
        for(index = 0; index <  rank; index++)
            running +=  A_private[index] * b[index*rank+c];
        output[r*rank + c] = running;
    }

    return;
}
