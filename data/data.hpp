#include <chrono>

#ifndef DATA_H_

	#define DATA_H_
	#include<stdint.h>
	//used to add a safety factor to the input/output arrays
	#ifndef SAFETY_FACTOR_SIZE
		#define SAFETY_FACTOR_SIZE 800
	#endif

	//The maximum number of iterations need to process the data
	#ifndef MAX_ITERATION_COUNT
		#define MAX_ITERATION_COUNT 100000
	#endif

	//maximum chunk size when joining relations larger than GPU memory
	#ifndef MAX_CHUNK_SIZE
		#define MAX_CHUNK_SIZE 128
	#endif

	#ifndef MILLION
		#define MILLION 1000000
	#endif

	//maximum match rate that will be tested
	#ifndef MAX_MATCH_RATE
		#define MAX_MATCH_RATE 8
	#endif

	//maximum output tuples that can fit within the GPU
	#ifndef MAX_OUTPUT_SIZE
		#define MAX_OUTPUT_SIZE (2 * MAX_CHUNK_SIZE * MILLION)
	#endif

	//maximum match rate that can be supported by my implementation
	#ifndef MAX_MATCH_RATE
		#define MAX_MATCH_RATE 20
	#endif

	//Warp size in CUDA
	#ifndef WARP_SIZE
		#define WARP_SIZE 32
	#endif

	//to allow easy data type change
	typedef unsigned long long data;

	struct Record {
		data key;
		data id;
	};

	typedef struct Record Record;

	//structure for storing the relation pointers and size
	struct relation_t{
		data *id; //pointer to the array containing id values
		data *key; //pointer to the array containing key values
		Record *records;
		long numTuples; //number of tuples in the relation
	};

	typedef struct relation_t relation_t;

	//Structure storing important arguments required for performing the hash join operation
	struct args_t{

		int *hRHisto[MAX_ITERATION_COUNT]; //Host side array for storing R relation histogram for each iteration.
		int *hSHisto[MAX_ITERATION_COUNT]; //Host side array for storing S relation histogram for each iteration.

		relation_t *hRelRn; //Host side array for storing partitioned R. Required only when R is larger than GPU memory.
		relation_t *hRelSn; //Host side array for storing partitioned S

		data *hOut[MAX_ITERATION_COUNT]; //Host side array for storing output of each iteration. Required only when S is larger than GPU memory.

		Record *hOutRcd[MAX_ITERATION_COUNT];

		int pCount; //partition Count
		int pCountL1; //partition count for level 1 partitioning. Used when shared memory is utilized.
		int pCountL2; //partition count for level 1 partitioning. Used when shared memory is utilized.

		int iterCount; //Number of iterations required to join the entire data set. Max relation size per iteration is 128M.

		int matchRate; //The match rate of the data set being generated

		float zFactor; //The z factor for generating the zipf relation

		std::chrono::steady_clock::time_point startTime, endTime; //variable for time measurement
	};

	typedef struct args_t args_t;
#endif