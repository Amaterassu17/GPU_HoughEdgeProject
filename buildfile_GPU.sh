#!/bin/bash

cd .
nvcc main_gpu.cu -o main_gpu
rm -rf ./output_GPU
# for i in {1..15}
# do
#     ./main_gpu triangle.png
# done

./main_gpu triangle.png

