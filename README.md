# matrix-multiplication-cuda

Modification of the code provided by Nvidia multiplying two matrices on a graphics card using CUDA. In the original code, one thread calculates one result, after modification, one thread can calculate K*L elements.
The files present a solution to the problem in two ways. In 'MatrixMul.cu', a single thread calculates adjacent elements, while in 'MatrixMul1.cu', elements are computed by one thread every size of K and L.
