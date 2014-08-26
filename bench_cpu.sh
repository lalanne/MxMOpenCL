#!/bin/bash

for wgSize in 1 2 4 8 16 32 64; do
    echo "Working Group Size ${wgSize} ${wgSize}"
    for i in {1..5}; do
        echo "execution number ${i}"
        ./mxm naive.cl ${wgSize} ${wgSize} >> wg_${wgSize}_${wgSize}_cpu.dat
    done
done

echo "Working Group Size 128 64"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 128 64 >> wg_128_64_cpu.dat
done

echo "Working Group Size 256 32"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 256 32 >> wg_256_32_cpu.dat
done

echo "Working Group Size 512 16"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 512 16 >> wg_512_16_cpu.dat
done

echo "Working Group Size 1024 1"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 1024 1 >> wg_1024_1_cpu.dat
done

echo "Working Group Size 64 128"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 64 128 >> wg_64_128_cpu.dat
done

echo "Working Group Size 32 256"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 32 256 >> wg_32_256_cpu.dat
done

echo "Working Group Size 16 512"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 16 512 >> wg_16_512_cpu.dat
done

echo "Working Group Size 1 1024"
for i in {1..5}; do
    echo "execution number ${i}"
    ./mxm naive.cl 1 1024 >> wg_1_1024_cpu.dat
done

