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

vector<float> mxm_host(const vector<float>& a, const vector<float>&b);
bool is_equal(const float x, const float y);
bool test_results(const vector<float>& a, 
                const vector<float>& b, 
                const vector<float>& results);
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

    if(test_results(a, b, c)) return SUCCESS;
    return EXIT_FAILURE;
}

vector<float> mxm_host(const vector<float>& a, const vector<float>&b){
    vector<float> swResults(DATA_SIZE);
    for(unsigned int i = 0; i < DATA_SIZE; i++){
        unsigned int row = i/MATRIX_RANK;
        unsigned int col = i%MATRIX_RANK;
        float running = 0.0f;

        for (unsigned int index=0;index<MATRIX_RANK;index++){
            unsigned int aIndex = row*MATRIX_RANK + index;
            unsigned int bIndex = col + index*MATRIX_RANK;
            running += a[aIndex] * b[bIndex];
        }
        swResults[i] = running;
    }
    return swResults;
} 

bool test_results(const vector<float>& a,
                        const vector<float>& b,
                        const vector<float>& results){
    unsigned int correct = 0;
    vector<float> swResults = mxm_host(a, b);

    for(unsigned int i = 0; i < DATA_SIZE; ++i){
        if(is_equal(swResults[i], results[i])) ++correct;
    }
    cout<<"Computed '"<<correct<<"/"<<DATA_SIZE<<"' correct values!"<<endl;
    if(correct == DATA_SIZE) return true;
    return false;
}

bool is_equal(const float x, const float y){
    const float epsilon = 1E-32;
    return abs(x - y) <= epsilon;
}

string load_program(string input){
    ifstream stream(input.c_str());
    if (!stream.is_open()) {
        cout << "Cannot open file: " << input << endl;
        exit(1);
    }
    return string(istreambuf_iterator<char>(stream), (istreambuf_iterator<char>()));
}


