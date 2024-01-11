#!/bin/bash

cd .
nvcc main_gpu.cu -o main_gpu_cu
./main_gpu_cu