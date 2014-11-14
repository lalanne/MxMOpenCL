#define __CL_ENABLE_EXCEPTIONS

#include <unistd.h>/*sleep*/
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>

#include "cl_noGL_lalanne.hpp"

const unsigned int MATRIX_RANK = 2;
const unsigned int DATA_SIZE = MATRIX_RANK*MATRIX_RANK;

#define DEVICE CL_DEVICE_TYPE_DEFAULT//CL_DEVICE_TYPE_CPU//CL_DEVICE_TYPE_GPU

using namespace std;
using namespace cl;

const unsigned int SUCCESS = 0;

int show_info(cl_platform_id platform_id);
unsigned int test_results(const float* const a, 
                        const float* const b, 
                        const float* const results);
string load_program(string input);

int main(int argc, char** argv){
    string kernel_name;
    if (argc != 2) { cout<<argv[0]<<" <inputfile>"<<endl; return EXIT_FAILURE; }
    else kernel_name = argv[1];

    vector<float> a(DATA_SIZE, 1.0);
    vector<float> b(DATA_SIZE, 1.0);

    Context context(DEVICE);
    CommandQueue queue(context);

    Buffer d_a(context, begin(a), end(a), true);
    Buffer d_b(context, begin(b), end(b), true);
    Buffer d_c(context, CL_MEM_WRITE_ONLY, DATA_SIZE * sizeof(float));
    
    string programText = load_program(kernel_name);
    try{
        Program program(context, programText, true);
        auto naive = make_kernel<Buffer, Buffer, Buffer>(program, "naive");
        NDRange global(MATRIX_RANK, MATRIX_RANK);
        naive(EnqueueArgs(queue, global), d_a, d_b, d_c);
        queue.finish();
    }
    catch(Error& e){
        cout<<"ERROR: exception: "<<e.what()<<" code: "<<e.err()<<endl;
        return EXIT_FAILURE;
    }

    vector<float> c(DATA_SIZE);
    try{
        cl::copy(queue, d_c, begin(c), end(c));
    }
    catch(Error& e){
        cout<<"ERROR: copy back exception: "<<e.what()<<" code: "<<e.err()<<endl;
        return EXIT_FAILURE;
    }

    std::copy(begin(c), end(c), ostream_iterator<float>(cout, ", "));
    cout<<endl<<endl;
    return SUCCESS;
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

string load_program(string input){
    ifstream stream(input.c_str());
    if (!stream.is_open()) {
        cout << "Cannot open file: " << input << endl;
        exit(1);
    }
    return string(istreambuf_iterator<char>(stream), (istreambuf_iterator<char>()));
}


