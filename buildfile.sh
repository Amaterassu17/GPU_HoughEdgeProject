#!/bin/bash

cd .
cd build
make
cd ..
cd bin
./GPU_HoughEdgeProject "../triangle.png"
cd ..