/*
 **********************************************
 *  CS314 Principles of Programming Languages *
 *  Fall 2020                                 *
 **********************************************
 */
#include "utils.hpp"
#include <strings.h>
#include "DataStructure.hpp"
#include "onewaywrapper.hpp"
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	char * filenameGraph = NULL;
	char * filenameOutput = (char *) "out.txt";
	int CUDA_device = 0;
	int num_threads = 16384;
	bool extra_credit = false;
	
	//examine command-line arguments:
    for(int arg = 1; arg < argc; arg++) {
		if(!strcmp(argv[arg], "-device")) {
			arg++;
			if(arg >= argc) {//bad arguments
				filenameGraph = NULL;
				break;
			}
			CUDA_device = atoi(argv[arg]);
		} else if(!strcmp(argv[arg], "-input")) {
			arg++;
			if(arg >= argc) {//bad arguments
				filenameGraph = NULL;
				break;
			}
			filenameGraph = argv[arg];
		} else if(!strcmp(argv[arg], "-output")) {
			arg++;
			if(arg >= argc) {//bad arguments
				filenameGraph = NULL;
				break;
			}
			filenameOutput = argv[arg];
		} else if(!strcmp(argv[arg], "-threads")) {
			arg++;
			if(arg >= argc) {//bad arguments
				filenameGraph = NULL;
				break;
			}
			num_threads = atoi(argv[arg]);
		} else if(!strcmp(argv[arg], "-extra")) {
			extra_credit = true;
		} else {
			filenameGraph = NULL;
			break;
		}
	}
	
	//if given invalid arguments, print usage info and quit:
	if (filenameGraph == NULL) {
        printUsage(argv[0]);
        exit(EXIT_FAILURE);
    }
	
	//set CUDA device to specified GPU:
	cudaSetDevice(CUDA_device);
	
	GraphData graph;
	
    //read the matrix/graph from the matrix market format file(.mtx) and sort it
    readmm(filenameGraph, &graph);

    //allocate memory for matching result
    int * res = (int *) malloc(graph.numNodes * sizeof(int));
	
    //initialize res to UNMATCHED
    for (int i = 0 ; i < graph.numNodes; i++) res[i] = -1;
	
    one_way_handshake_wrapper(graph, res, num_threads, extra_credit);
	
    //write result to output file
    write_match_result(filenameOutput, res, graph.numNodes);

    //clean allocated memory
	free(res);
	free(graph.src);
	free(graph.dst);
	free(graph.weight);

    return 0;
}
