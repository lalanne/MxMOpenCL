
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel //__attribute__ ((reqd_work_group_size(16, 16, 1)))
void naive(__global double* a, __global double* b, __global double* output)
{
  int r = get_global_id(0);
  int c = get_global_id(1);
  int rank = get_global_size(0);
  double running = 0.0f;

  for (int index=0; index<rank; index++) {
    int aIndex = r*rank + index;
    int bIndex = index*rank + c;
    running +=  a[aIndex] * b[bIndex];
  }
  
  output[r*rank + c] = running;
  return;
}
