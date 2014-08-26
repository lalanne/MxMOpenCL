#!/bin/bash

for wgSize in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
    echo "Working Group Size ${wgSize}"
    for i in {1..5}; do
        echo "execution number ${i}"
        ./mxm_gpu private_memory.cl ${wgSize} >> wg_${wgSize}_gpu.dat
    done
done
