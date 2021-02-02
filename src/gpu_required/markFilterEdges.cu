/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Fall 2020                                 *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void markFilterEdges_gpu(int * src, int * dst, int * matches, int * keepEdges, int numEdges) {
	/** YOUR CODE GOES BELOW **/
    int threadNum = blockDim.x * gridDim.x;
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = threadId; i < numEdges; i += threadNum) {
        keepEdges[i] = (matches[src[i]] != -1 || matches[dst[i]] != -1) ? 0 : 1;
    }
	/** YOUR CODE GOES ABOVE **/
}
