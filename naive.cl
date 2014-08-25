
__kernel //__attribute__ ((reqd_work_group_size(16, 16, 1)))
void naive(__global float* a, __global float* b, __global float* output)
{
  int r = get_global_id(0);
  int c = get_global_id(1);
  int rank = get_global_size(0);
  float running = 0.0f;

  for (int index=0; index<rank; index++) {
    int aIndex = r*rank + index;
    int bIndex = index*rank + c;
    running +=  a[aIndex] * b[bIndex];
  }
  
  output[r*rank + c] = running;
  return;
}
