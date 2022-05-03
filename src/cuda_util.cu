#include "cuda_util.h"
#include "device_launch_parameters.h"

dim3 compute_grid_size(int tblock_count)
{
	dim3 size(tblock_count);
	while (size.x > 0xffff) {
		size.x = (size.x + 1) >> 1;
		size.y <<= 1;
	}
	return size;
}