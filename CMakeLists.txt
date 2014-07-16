
cmake_minimum_required (VERSION 2.6) 
project (mxm) 

#add_definitions(-framework OpenCL)
add_definitions(-Wall)

add_executable(mxm
            test-cl.c)

