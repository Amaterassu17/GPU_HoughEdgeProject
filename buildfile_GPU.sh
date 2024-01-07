#!/bin/bash

cd .
nvcc main_gpu_cu.cu -o main_gpu_cu
./main_gpu_cu