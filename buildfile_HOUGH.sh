#!/bin/bash

cd .
cd build_hough
make
cd ..
cd bin
./GPU_HoughEdgeProject
cd ..