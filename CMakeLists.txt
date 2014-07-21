
cmake_minimum_required (VERSION 2.6) 
project (mxm) 

#add_definitions(-framework OpenCL)
add_definitions(-Wall)

add_executable(mxm
            test-cl.c)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(mxm OpenCL)
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")
