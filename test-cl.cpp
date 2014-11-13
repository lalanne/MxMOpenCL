#include <unistd.h>/*sleep*/
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "cl.hpp"

#define MATRIX_RANK 4
#define DATA_SIZE MATRIX_RANK*MATRIX_RANK

#define DEVICE CL_DEVICE_TYPE_CPU//CL_DEVICE_TYPE_GPU

using namespace std;
using namespace cl;

const unsigned int SUCCESS = 0;

int show_info(cl_platform_id platform_id);
int load_file_to_memory(const char *filename, char **result);
unsigned int test_results(const float* const a, 
                        const float* const b, 
                        const float* const results);
string loadProgram(string input);

int main(int argc, char** argv){
    int err;                            
     
    vector<float> a(DATA_SIZE);
    vector<float> b(DATA_SIZE);
    vector<float> c(DATA_SIZE);

    Context context(DEVICE);
    CommandQueue queue(context);

    Buffer d_a, d_b, d_c;
   
    if (argc != 4){
        printf("%s <inputfile>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const unsigned int wgSize1D = atoi(argv[2]);
    const unsigned int wgSize2D = atoi(argv[3]);
    printf("Working Group size 1D[%u] 2D[%u] kernel[%s] \n", wgSize1D, 
                                                            wgSize2D, 
                                                            argv[1]);
    // Fill our data sets with pattern
    int i;
    for(i = 0; i < DATA_SIZE; i++){
        a[i] = (float)i;
        b[i] = (float)i;
        c[i] = 0.0f;
    }

    d_a = Buffer(context, begin(a), end(a), true);
    d_b = Buffer(context, begin(b), end(b), true);
    d_c = Buffer(context, begin(c), end(c), true);

    try{
        Program program(context, loadProgram("naive.cl"), true);
        auto naive = make_kernel<float, float, float, Buffer, Buffer, Buffer>(program, "naive");

        NDRange global(MATRIX_RANK, MATRIX_RANK);
        naive(EnqueueArgs(queue, global), MATRIX_RANK, MATRIX_RANK, MATRIX_RANK, d_a, d_b, d_c);
    }
    catch(...){
        cout<<"Exception!!!"<<endl;
    }

    queue.finish();
    sleep(3);

    copy(queue, d_c, begin(c), end(c));
    cout<<"here"<<endl;
}

unsigned int test_results(const float* const a,
                        const float* const b,
                        const float* const results){
    unsigned int correct = 0;
    float sw_results[DATA_SIZE];          // results returned from device

    unsigned int i;
    for(i = 0; i < DATA_SIZE; i++){
        int row = i/MATRIX_RANK;
        int col = i%MATRIX_RANK;
        float running = 0.0f;

        int index;
        for (index=0;index<MATRIX_RANK;index++){
            int aIndex = row*MATRIX_RANK + index;
            int bIndex = col + index*MATRIX_RANK;
            running += a[aIndex] * b[bIndex];
        }
        sw_results[i] = running;
    }

    for (i = 0;i < DATA_SIZE; i++) {
        if(abs(results[i] - sw_results[i]) < 1E-32){
            correct++;
        }
    }

    printf("Computed '%d/%d' correct values!\n", correct, DATA_SIZE);
    return correct;
}

int show_info(cl_platform_id platform_id){
    int err;
    char cl_platform_vendor[1001];
    char cl_platform_name[1001];
    char cl_platform_version[1001];

    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VENDOR,1000,(void *)cl_platform_vendor,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VENDOR %s\n",cl_platform_vendor);
  
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME,1000,(void *)cl_platform_name,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_NAME) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_NAME %s\n",cl_platform_name);
    
    err = clGetPlatformInfo(platform_id,CL_PLATFORM_VERSION,1000,(void *)cl_platform_version,NULL);
    if (err != CL_SUCCESS){
        printf("Error: clGetPlatformInfo(CL_PLATFORM_VERSION) failed!\n");
        printf("Test failed\n");
        return EXIT_FAILURE;
    }
    printf("CL_PLATFORM_VERSION %s\n",cl_platform_version);

    return SUCCESS;
}

int load_file_to_memory(const char *filename, char **result){ 
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL){ 
        *result = NULL;
        return -1; // -1 means file opening fail 
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f)){ 
        free(*result);
        return -2; // -2 means file reading fail 
    }  
    fclose(f);
    (*result)[size] = 0;
  
    return size;
}

string loadProgram(string input){
    ifstream stream(input.c_str());
    if (!stream.is_open()) {
        cout << "Cannot open file: " << input << endl;
        exit(1);
    }

    cout<<string(istreambuf_iterator<char>(stream), (istreambuf_iterator<char>()))<<endl;

     return string(
        istreambuf_iterator<char>(stream),
        (istreambuf_iterator<char>()));
}


