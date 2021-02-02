/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Fall 2020                                 *
 **********************************************
 */
#include <stdio.h>
#include <stdlib.h>

__global__ void packGraph_gpu(int * newSrc, int * oldSrc, int * newDst, int * oldDst, int * newWeight, int * oldWeight, int * edgeMap, int numEdges) {
	//Get current thread ID and total number of threads in grid
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	
	for(int x = tid; x < numEdges; x += num_threads) {
		int myMap = edgeMap[x];
		int nextMap = edgeMap[x + 1];
		
		if(myMap != nextMap) {
			newSrc[myMap] = oldSrc[x];
			newDst[myMap] = oldDst[x];
			newWeight[myMap] = oldWeight[x];
		}
	}
}
