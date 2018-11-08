#ifndef DATA_HPP_
#define DATA_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

//maximum number of cuda streams
#ifndef STREAM_COUNT
	#define STREAM_COUNT 256
#endif

//maximum number of threads inside a block
#ifndef MAX_BLOCK_SIZE
	#define MAX_BLOCK_SIZE 1024
#endif

//maximum number of block inside a grid
#ifndef MAX_GRID_SIZE
	#define MAX_GRID_SIZE 55535
#endif

//grid size multiplier for level 2 partitioning
#ifndef GRID_SIZE_MULTIPLIER
	#define GRID_SIZE_MULTIPLIER 40
#endif

//this structure stores important CUDA parameters
struct cudaParameters_t{
	cudaStream_t streams[STREAM_COUNT];
	int gridSize;
	int blockSize;
	int tlp; //tile size for partition phase
	int tlj; //tile size for probe phase
	int gridSizeMultiplier;
	int sharedMemSize;
	cudaEvent_t start, stop;
	float time;
};

typedef struct cudaParameters_t cudaParameters_t;

#endif