
__kernel
void naive(__global float* a, __global float* b, __global float* output)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    int rank = get_global_size(0);
    float running = 0.0f;

    //printf("r[%d].......\n", r);
    //printf("c[%d].......\n", c);
    //printf("rank[%d].......\n", rank);
    //printf("\n");

    for (int index=0; index<rank; index++) {
        int aIndex = r*rank + index;
        int bIndex = index*rank + c;
        running +=  a[aIndex] * b[bIndex];
    }
  
    output[r*rank + c] = running;
}
