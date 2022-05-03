#pragma once
#include "device_launch_parameters.h"
#include <cassert>

#define THREADCOUNT 256

#define CUDA_CHECK(err) \
    {\
        if( cudaSuccess != err) { \
           fprintf(stderr, "[GPUJPEG] [Error] %s (line %i): %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			throw(std::exception(cudaGetErrorString(err)));\
        } \
    } \

#define div_and_round_up(value, div) \
    (((value % div) != 0) ? (value / div + 1) : (value / div))

dim3 compute_grid_size(int tblock_count);