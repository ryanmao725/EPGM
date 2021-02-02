/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Fall 2020                                 *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void exclusive_prefix_sum_gpu(int * oldSum, int * newSum, int distance, int numElements) {
	/** YOUR CODE GOES BELOW **/
    int threadNum = blockDim.x * gridDim.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    for (int i = threadId; i < numElements; i += threadNum) {
        if (distance == 0) {
            newSum[0] = 0;
            if (i < numElements - 1) {
                newSum[i + 1] = oldSum[i];
            }
        } else {
            if (i >= distance) {
                newSum[i] = oldSum[i] + oldSum[i - distance];
            }
            else {
                newSum[i] = oldSum[i];
            }
        }
    }
	/** YOUR CODE GOES ABOVE **/
}
