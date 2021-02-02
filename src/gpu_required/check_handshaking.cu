/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Fall 2020                                 *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void check_handshaking_gpu(int * strongNeighbor, int * matches, int numNodes) {
	/** YOUR CODE GOES BELOW **/
    // Get thread ID ( for a 1 dimensional block and grid )
    int threadNum = blockDim.x * gridDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int i = threadId; i < numNodes; i += threadNum) {
        if (matches[i] == -1) {
            if (strongNeighbor[i] != -1 && strongNeighbor[strongNeighbor[i]] == i) {
                matches[i] = strongNeighbor[i];
            }
        }
    }
	/** YOUR CODE GOES ABOVE **/
}
