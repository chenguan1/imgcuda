
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "cuda_util.h"


__global__ void kernel_test()
{
    
}


void cuda_test()
{
	int data_size = 2448 * 2048;

	cudaSetDevice(0);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);


	// º∆À„ Block Size ∫Õ Grid Size
	dim3 Db(THREADCOUNT); //Db.x==THREADCOUNT, Db.y==1, Db.z==1
	int blockCount = div_and_round_up(data_size, Db.x);
	dim3 Dg = compute_grid_size(blockCount);

	kernel_test <<<Dg, Db>>>();

	auto err = cudaGetLastError();
	printf("%s", cudaGetErrorString(err));

	getchar();
}