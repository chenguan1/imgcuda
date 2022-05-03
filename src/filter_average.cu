
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_util.h"

#include <string>
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


__global__ void kernel_filter_average(uchar *data_in, uchar* data_out, int height, int width, int pitch, int radius)
{
	int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
	int thread_id = block_idx * blockDim.x + threadIdx.x;
	int image_size = pitch * height;
	if (thread_id >= image_size) return;

	int x = thread_id % pitch;
	int y = thread_id / pitch;

	if (x >= width) return;

	int x0, y0;

	int sum_value = 0;
	int sum_count = 0;
	for (y0 = y - radius; y0 <= y + radius; y0++) {
		if (y0 < 0 || y0 >= height) continue;
		for (x0 = x - radius; x0 <= x + radius; x0++) {
			if (x0 < 0 || x0 >= width) continue;
			++sum_count;
			sum_value += data_in[y0 * pitch + x0];
		}
	}

	data_out[y*pitch +x] = (uchar)(sum_value / sum_count);
}



int filter_average()
{
	uchar* data_dev = nullptr;
	uchar* data_dev_out = nullptr;
	size_t data_size;
	int width, height;
	size_t pitch = 0;

	int ksize = 5;

	string path = "img/blood.bmp";
	string path_out_cv = "img/blood_average_cv.bmp";
	string path_out_cuda = "img/blood_average_cuda.bmp";
	string path_out_tex = "img/blood_average_tex.bmp";


	Mat img = imread(path, 0);
	Mat img_out(img.rows, img.cols, img.type());

	width = img.cols;
	height = img.rows;
	

	//////////////////////////////////////////////////////////////////
	// by opencv
	double timex = static_cast<double>(getTickCount());
	blur(img, img_out, Size(ksize, ksize));
	timex = ((double)getTickCount() - timex) / getTickFrequency();
	cout << "[opencv] time spend " << timex * 1000 << " ms" << endl;
	imwrite(path_out_cv, img_out);
	img_out.setTo(0);

	/////////////////////////////////////////////////////////////////
	// by cuda
	CUDA_CHECK(cudaMallocPitch(&data_dev, &pitch, width, height));
	CUDA_CHECK(cudaMallocPitch(&data_dev_out, &pitch, width, height));
	CUDA_CHECK(cudaMemcpy2D(data_dev, pitch, img.data, width, width, height, cudaMemcpyHostToDevice));

	//pitch = width;
	//CUDA_CHECK(cudaMalloc(&data_dev, data_size));
	//CUDA_CHECK(cudaMalloc(&data_dev_out, data_size));
	//CUDA_CHECK(cudaMemcpy(data_dev, img.data, data_size, cudaMemcpyHostToDevice));

	data_size = pitch * height;

	
	// 计算 Block Size 和 Grid Size
	dim3 Db(THREADCOUNT); //Db.x==THREADCOUNT, Db.y==1, Db.z==1
	int blockCount = div_and_round_up(data_size, Db.x);
	dim3 Dg = compute_grid_size(blockCount);

	// 计时
	cudaEvent_t ev_start, ev_end;
	CUDA_CHECK(cudaEventCreate(&ev_start));
	CUDA_CHECK(cudaEventCreate(&ev_end));


	// 开始执行
	CUDA_CHECK(cudaEventRecord(ev_start));
	kernel_filter_average<<<Dg, Db>>>(data_dev, data_dev_out, height, width, pitch, ksize / 2);
	auto err = cudaGetLastError();
	CUDA_CHECK(err);
	CUDA_CHECK(cudaEventRecord(ev_end));
	CUDA_CHECK(cudaEventSynchronize(ev_end));
	
	// 打印时间
	float fms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&fms, ev_start, ev_end));
	cout << "[cuda] time spend " << fms << " ms" << endl;

	// 保存处理结果
	CUDA_CHECK(cudaMemcpy2D(img_out.data, width, data_dev_out, pitch, width, height, cudaMemcpyDeviceToHost));
	//CUDA_CHECK(cudaMemcpy(img_out.data, data_dev_out, data_size, cudaMemcpyDeviceToHost));
	imwrite(path_out_cuda, img_out);


	// 清理
	CUDA_CHECK(cudaEventDestroy(ev_end));
	CUDA_CHECK(cudaEventDestroy(ev_start));
	CUDA_CHECK(cudaFree(data_dev));
	CUDA_CHECK(cudaFree(data_dev_out));

	cin.ignore();

	return 0;
}