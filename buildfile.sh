#!/bin/bash

cd .
cd build
make
cd ..
cd bin

# Run the program 15 times
# for i in {1..15}
# do
#     ./GPU_HoughEdgeProject "../triangle.png"
# done

./GPU_HoughEdgeProject "../triangle.png"


# ./GPU_HoughEdgeProject "../triangle.png"
cd ..