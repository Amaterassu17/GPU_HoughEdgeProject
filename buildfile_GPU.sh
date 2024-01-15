#!/bin/bash

cd .
nvcc main_gpu.cu -o main_gpu
./main_gpu