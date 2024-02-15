#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <assert.h>
#include <utility>


#define cdpErrchk(ans) { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      printf("GPU kernel assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) assert(0);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16
#define TILE_SIZE 16


#define LAPLACIAN_GAUSSIAN 0
#define GAUSSIAN_KERNEL_SIZE 5
#define GAUSSIAN_SIGMA 2.0
#define SHARED 0
#define TILED 0
#define HYS_STACK 1


#define MAX_THRESHOLD_MULT 0.4//*255
#define MIN_THRESHOLD_MULT 0.005 //*255
#define NON_MAX_SUPPR_THRESHOLD 0.8
#define HOUGH_THRESHOLD_MULT 0.7


__constant__ float sobel_h_constant[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float sobel_v_constant[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
__constant__ float gaussian_filter_constant[GAUSSIAN_KERNEL_SIZE*GAUSSIAN_KERNEL_SIZE];



__global__ void apply_filter_global(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < height && j < width){
		float sum = 0.0f; // Initialize sum to 0 inside the loop

		for (int k = 0; k < kernel_size; k++)
		{
			for (int m = 0; m < kernel_size; m++)
			{
				int input_row = i + (k - 1);
				int input_col = j + (m - 1);

				// Check if the indices are within bounds
				if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
					sum += kernel[k * kernel_size + m] * input[input_row * width + input_col];
				}	
			}
		}

		output[i * width + j] = abs(sum);
	}
}


__global__ void apply_filter_shared(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	extern __shared__ float kernel_shared[];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
		kernel_shared[threadIdx.y * kernel_size + threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
	}

	__syncthreads(); // Ensure all threads have finished copying to shared memory

	if (i < height && j < width) {
		float sum = 0;
		for (int k = 0; k < kernel_size; k++) {
			for (int m = 0; m < kernel_size; m++) {
				int input_row = i + (k - 1);
				int input_col = j + (m - 1);

				// Check if the indices are within bounds
				if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
					sum += kernel_shared[k * kernel_size + m] * input[input_row * width + input_col];
				}
			}
		}	

		output[i * width + j] = abs(sum);
	}
}


//0 gaussian
//1 sobel_h
//2 sobel_v
__global__ void apply_filter_shared_tiled(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, int code){
	
	__shared__ uint8_t input_shared[TILE_SIZE][TILE_SIZE];

	float* kernel_shared;
	if(code == 0){
		kernel_shared= gaussian_filter_constant;
		
	} else if(code == 1){
		kernel_shared= sobel_h_constant;
	} else if(code == 2){
		kernel_shared= sobel_v_constant;
	}


	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < height && j < width) {
		input_shared[threadIdx.y+(kernel_size/2)][threadIdx.x+(kernel_size/2)] = input[i * width + j];
		//Manage tiling if on the borders adding padding on the borders and the other values of the matrix if the border is inner
		if(i==0 && j==0)
		{
			for(int k = 0; k < kernel_size/2; k++){
				for(int m = 0; m < kernel_size/2; m++){
					input_shared[k][m] = 0;
				}
			}
		}
		else if(i==0 && j==width-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				for(int m = 0; m < kernel_size/2; m++){
					input_shared[k][TILE_SIZE-1-m] = 0;
				}
			}
		}
		else if(i==height-1 && j==0)
		{
			for(int k = 0; k < kernel_size/2; k++){
				for(int m = 0; m < kernel_size/2; m++){
					input_shared[TILE_SIZE-1-k][m] = 0;
				}
			}
		}
		else if(i==height-1 && j==width-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				for(int m = 0; m < kernel_size/2; m++){
					input_shared[TILE_SIZE-1-k][TILE_SIZE-1-m] = 0;
				}
			}
		}
		else if(i==0)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[k][threadIdx.x] = 0;
			}
		}
		else if(i==height-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[TILE_SIZE-1-k][threadIdx.x] = 0;
			}
		}
		else if(j==0)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[threadIdx.y][k] = 0;
			}
		}
		else if(j==width-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[threadIdx.y][TILE_SIZE-1-k] = 0;
			}
		}
		
		if(threadIdx.x == TILE_SIZE-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[threadIdx.y][TILE_SIZE-1+k] = input[i*width + j+k];
			}
		}
		if(threadIdx.y == TILE_SIZE-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				input_shared[TILE_SIZE-1+k][threadIdx.x] = input[(i+k)*width + j];
			}
		}
		if(threadIdx.x == TILE_SIZE-1 && threadIdx.y == TILE_SIZE-1)
		{
			for(int k = 0; k < kernel_size/2; k++){
				for(int m = 0; m < kernel_size/2; m++){
					input_shared[TILE_SIZE-1+k][TILE_SIZE-1+m] = input[(i+k)*width + j+m];
				}
			}
		}
	}

	__syncthreads(); // Ensure all threads have finished copying to shared memory

	if(i==0 && j==0){
		for(int k = 0; k < TILE_SIZE+kernel_size-1; k++){
			for(int m = 0; m < TILE_SIZE+kernel_size-1; m++){
				printf("%d, ", input_shared[k][m]);
			}
		}
	}

	if (i < height && j < width) {
		float sum = 0;


		for (int k = 0; k < kernel_size; k++) {
			for (int m = 0; m < kernel_size; m++) {
				int input_row = i + (k - 1);
				int input_col = j + (m - 1);

				// Check if the indices are within bounds
				if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
					sum += kernel_shared[k * kernel_size + m] * input_shared[threadIdx.y + (k - 1)][threadIdx.x + (m - 1)];
					if(threadIdx.x == 1 && threadIdx.y == 1){
						//printf("k = %d, m= %d, kernel_shared -> %f, input -> %d\n",k,m, kernel_shared[k * kernel_size + m], input[input_row * width + input_col]);
					}
				}
			}
		}
		

		output[i * width + j] = abs(sum);
	}


}

__global__ void convert_to_greyscale(int height, int width, uint8_t *img, uint8_t *grey_img)
{

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < height && j < width){
		auto b = img[i*width*3 + j*3 + 0];
		auto g = img[i*width*3 + j*3 + 1];
		auto r = img[i*width*3 + j*3 + 2];

		int average = (int)(0.3*r + 0.59*g + 0.11*b); // Adjust the weights for each channel

		grey_img[i*width + j] = average;
	}
}

/**
 * Compute magnitude and gradient
*/

__global__ void compute_magnitude_and_gradient(int height, int width, uint8_t *Ix, uint8_t *Iy, uint8_t *mag, float *grad){


	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < height && j < width){
		float dx = Ix[i*width+j];
		float dy = Iy[i*width+j];
		mag[i*width+j] = round(sqrt(dx*dx+dy*dy));
		float angle = atan2(dy, dx)*180/M_PI;
		grad[i*width+j] = angle < 0 ? angle+180 : angle;
	}
	

}

/**
 * NON MAXIMUM SUPPRESSION

*/

__global__ void non_maximum_suppression_non_interpolated(int height, int width, uint8_t *suppr_mag, uint8_t *mag, float* grad){

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	// here shared memory could be used to save some global loads from mag

	if(i<height && j<width){
		int q = 255;
		int r = 255;

		float grad_ij = grad[i*width+j];

		//angle 0
		if (0 <= grad_ij < 22.5 || 157.5 <= grad_ij <= 180){
			q = mag[i*width + j+1];
			r = mag[i*width + j-1];
		}
		//angle 45
		else if (22.5 <= grad_ij < 67.5){
			q = mag[(i+1)*width + j-1];
			r = mag[(i-1)*width + j+1];
		}
		//angle 90
		else if (67.5 <= grad_ij < 112.5){
			q = mag[(i+1)*width + j];
			r = mag[(i-1)*width + j];
		}
		//angle 135
		else if (112.5 <= grad_ij < 157.5){
			q = mag[(i-1)*width + j-1];
			r = mag[(i+1)*width + j+1];
		}

		float mag_ij = mag[i*width + j];

		if (mag_ij >= q*NON_MAX_SUPPR_THRESHOLD && mag_ij >= r * NON_MAX_SUPPR_THRESHOLD){
			suppr_mag[i*width + j] = mag_ij;
		} else {
			suppr_mag[i*width + j] = 0;
		}
	}


}

/**
 * THRESHOLD
*/

__global__ void float_threshold(int height, int width,  uint8_t *pixel_classification,  uint8_t *suppr_mag, float max_mag){

	float high_threshold = MAX_THRESHOLD_MULT*max_mag;
	float low_threshold = MIN_THRESHOLD_MULT*max_mag;


	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < height && j < width){
		if(suppr_mag[i*width+j] >= high_threshold){
			// strong pixels
			pixel_classification[i*width+j] = 255;
		} else if (suppr_mag[i*width+j] < low_threshold){
			// non relevant pixels
			pixel_classification[i*width+j] = 0;
		} else {
			// weak pixels
			pixel_classification[i*width+j] = 50;
		}
	}
}

/**
 * HYSTERESIS
*/

__global__ void hysteresis(int height, int width, uint8_t *pixel_classification){

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;


	// printf("i = %d, j = %d\n", i, j);
	if(i < height && j < width){
		if(pixel_classification[i*width+j] == 50){
			if(pixel_classification[(i+1)*width+j-1] == 255 || pixel_classification[(i+1)*width+j] == 255 || pixel_classification[(i+1)*width+j+1] == 255 ||
			pixel_classification[i*width+j-1] == 255 || pixel_classification[i*width+j+1] == 255 || pixel_classification[(i-1)*width+j-1] == 255 ||
			pixel_classification[(i-1)*width+j] == 255 || pixel_classification[(i-1)*width+j+1] == 255){
				pixel_classification[i*width + j] = 255;
			} else {
				pixel_classification[i*width + j] = 0;
			}
		}
	}

}

__global__ void hysteresis_stack(int height, int width, uint8_t *pixel_classification){
		
	extern __shared__ int stack_sdata[];
	__shared__ int stack_depth;

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(threadIdx.x == 0 && threadIdx.y == 0)
		stack_depth = 0;
	__syncthreads();

	// pixel is a strong edge pixel
	if(i < height && j < width && pixel_classification[i*width+j] == 255){
		for(int l=-1; l<2; l++){
			for(int m=-1; m<2; m++){
				int index = (i+l)*width+(j+m);
				if((i+l) > 0 && (j+m) > 0 && (i+l) < height & (j+m) < width &&
					pixel_classification[index] == 50){
					// neighbour is weak edge pixel -> push to stack
					int tmp_stack_depth = atomicAdd(&stack_depth, 1);
					stack_sdata[tmp_stack_depth] = index;
				}
			}
		}
	}
	
	__syncthreads();
	
	while(stack_depth > 0){

		if(threadIdx.y * blockDim.x + threadIdx.x < stack_depth){
			// tmp_stack_depth stores value before 1 was subtracted
			int tmp_stack_depth = atomicSub(&stack_depth, 1);
			if(tmp_stack_depth > 0){
				// every thread pops a different element
				int indexx = stack_sdata[tmp_stack_depth-1];
				// set current one to final edge pixel
				pixel_classification[indexx] = 255;
				int i_top = floor((indexx)*1.0 / width);
				int j_top = indexx - i_top*width;
				for(int l=-1; l<2; l++){
					for(int m=-1; m<2; m++){
						int new_idx = (i_top+l)*width+(j_top+m);
						if(m != 0 && l !=0 && (i_top+l) > 0 && (j_top+m) > 0 && (i_top+l) < height & (j_top+m) < width &&
						pixel_classification[new_idx] == 50){
							// neighbour is valid weak edge pixel -> push to stack
							int tmp_stack_depth = atomicAdd(&stack_depth, 1);
							stack_sdata[tmp_stack_depth] = new_idx;
						}
					}
				}
			}
		}
	}

	__syncthreads();

	// remove remaining weak edge pixels
	if(i < height && j < width && pixel_classification[i*width+j] == 50){
		pixel_classification[i*width+j] = 0;
	}
}

/**
 * DILATION AND EROSION
*/

__global__ void apply_dilation_global(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < height - 1 && j >= 1 && j < width - 1) {
        uint8_t max_val = 0;
        for (int k = 0; k < kernel_size; k++) {
            for (int m = 0; m < kernel_size; m++) {
                // Ensure valid indices for input array
                int input_index = (i + (k - 1)) * width + j + (m - 1);
                    auto value = kernel[k * kernel_size + m] * input[input_index];
                    max_val = max_val > value ? max_val : value;
                
            }
        }
        output[i * width + j] = abs(max_val);
    }
}


__global__ void apply_erosion_global(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

if(i < height && j < width){
	uint8_t min_val = 255;
	for (int k = 0; k < kernel_size; k++) {
		for (int m = 0; m < kernel_size; m++) {

			min_val = min_val < kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)] ? min_val : kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)];

		}
	}
	output[i * width + j] = abs(min_val);

}
}

__global__ void apply_dilation_shared(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
	extern __shared__ float kernel_shared[];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
		kernel_shared[threadIdx.y * kernel_size + threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
	}

	__syncthreads(); // Ensure all threads have finished copying to shared memory

	if (i >= 1 && i < height - 1 && j >= 1 && j < width - 1) {
		uint8_t max_val = 0;
		for (int k = 0; k < kernel_size; k++) {
			for (int m = 0; m < kernel_size; m++) {
				// Ensure valid indices for input array
				int input_index = (i + (k - 1)) * width + j + (m - 1);
				auto value = kernel_shared[k * kernel_size + m] * input[input_index];
				max_val = max_val > value ? max_val : value;
			}
		}
		output[i * width + j] = abs(max_val);
	}
}

__global__ void apply_erosion_shared(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
	extern __shared__ float kernel_shared[];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
		kernel_shared[threadIdx.y * kernel_size + threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
	}

	__syncthreads(); // Ensure all threads have finished copying to shared memory

	if (i >= 1 && i < height - 1 && j >= 1 && j < width - 1) {
		uint8_t min_val = 255;
		for (int k = 0; k < kernel_size; k++) {
			for (int m = 0; m < kernel_size; m++) {
				// Ensure valid indices for input array
				int input_index = (i + (k - 1)) * width + j + (m - 1);
				min_val = min_val < kernel_shared[k * kernel_size + m] * input[input_index] ? min_val : kernel_shared[k * kernel_size + m] * input[input_index];
			}
		}
		output[i * width + j] = abs(min_val);
	}
}

/**
 * HOUGH TRANSFORM
 
*/

__global__ void hough_transform(int height, int width, int max_rho, int max_theta, float threshold_mult, uint8_t* img, int* hough_space, uint8_t *output, int channels, int* lines){

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width && img[i * width + j] > 0) {
        for (int thetaIdx = 0; thetaIdx < max_theta; thetaIdx++) {
            double theta = thetaIdx * M_PI / max_theta;
            double rho = j * cos(theta) + i * sin(theta);
            int rhoIdx = (int)(rho + max_rho / 2);

            if (rhoIdx >= 0 && rhoIdx < max_rho) {
                atomicAdd(&hough_space[rhoIdx * max_theta + thetaIdx], 1);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int maxVote = 0;
        for (int k = 0; k < max_rho; k++) {
            for (int m = 0; m < max_theta; m++) {
                int vote = hough_space[k * max_theta + m];
                if (vote > maxVote) {
                    maxVote = vote;
                    lines[0] = k;
                    lines[1] = m;
                }
            }
        }
    }

    __syncthreads();

    if (i < height && j < width) {
        if (hough_space[lines[0] * max_theta + lines[1]] > 255 * threshold_mult) {
            double rho = lines[0] - max_rho / 2.0;
            double theta = lines[1] * M_PI / max_theta;

            for (int xx = 0; xx < width; xx++) {
                int yy = (int)((rho - xx * cos(theta)) / sin(theta));
                if (yy >= 0 && yy < height) {
                    output[(yy * width + xx) * channels] = 255;
                    output[(yy * width + xx) * channels + 1] = 255;
                    output[(yy * width + xx) * channels + 2] = 0;
                }
            }
        }
    }
}


//UTILITY

float* get_gaussian_filter (int kernel_size, float sigma){

	kernel_size = kernel_size%2 == 0 ? kernel_size-1 : kernel_size;

	float* gaussian_filter = (float*)malloc(kernel_size*kernel_size*sizeof(float));
	float sum = 0.0;
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			gaussian_filter[i*kernel_size + j] = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
			// gaussian_filter_constant[i*kernel_size + j] = gaussian_filter[i*kernel_size + j];
			sum += gaussian_filter[i*kernel_size + j];
		}
	}
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			gaussian_filter[i*kernel_size + j] /= sum;
			//gaussian_filter_constant[i*kernel_size + j] /= sum;
		}
	}
	return gaussian_filter;

}

float* get_gaussian_laplacian_filter (int kernel_size, float sigma){
	float* gaussian_filter = (float*)malloc(kernel_size*kernel_size*sizeof(float));
	float sum = 0.0;
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			gaussian_filter[i*kernel_size + j] = (((i*i+j*j)/(2*sigma*sigma))-1)*exp(-(i*i+j*j)/(2*sigma*sigma))/(M_PI*sigma*sigma*sigma*sigma);
			sum += gaussian_filter[i*kernel_size + j];
		}
	}
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			gaussian_filter[i*kernel_size + j] /= sum;
		}
	}
	return gaussian_filter;


}



int main(int argc, char *argv[])
{
    const int blocksize = BLOCK_SIZE;
    int device;
    struct cudaDeviceProp properties;
    
    cudaError_t err = cudaSuccess;
    cudaDeviceProp deviceProp;
    int devID = 0;
    auto error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_FAILURE);
    }

    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }


    dim3 threads, grid;
    
    //image definitions
    int width, height, bpp;
    
	auto img_fname = argc>=2 ? argv[1] : "image.png";

	system("mkdir -p output_GPU");


	auto file_times = fopen("./output_GPU/times_Canny_GPU.csv", "a");

	//put headers
	fprintf(file_times, "convert_to_greyscale,apply_gaussian_filter,apply_sobel_filters,compute_magnitude_and_gradient,non_maximum_suppression,double_threshold,hysteresis,dilation,erosion,Hough Transform, TotalTime\n");
	

    //program starts

	uint8_t* rgb_image = stbi_load(img_fname, &width, &height, &bpp, 3);
    uint8_t* rgb_image_d;
    cudaMalloc(&rgb_image_d, width*height*3);
    cudaMemcpy(rgb_image_d, rgb_image, width*height*3, cudaMemcpyHostToDevice);

	threads = dim3(blocksize, blocksize);
    grid = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    printf("CUDA kernel launch with %d blocks of %d threads\n", grid.x * grid.y, threads.x * threads.y);



    std::cout<<"image: "<<img_fname<<std::endl;
	std::cout<<width<<" "<<height<<std::endl;


	cudaEvent_t start, stop, start_total, stop_total;
  	float msecTotal;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	cudaEventCreate(&start_total);
	cudaEventCreate(&stop_total);

	cudaEventRecord(start_total, NULL);	

    //Stop here



	// Convert to greyscale
	uint8_t* grey_image;
	uint8_t* grey_image_d;
	
	cudaEventRecord(start, NULL);

	grey_image = (uint8_t*)malloc(width*height);


	cudaMalloc(&grey_image_d, width*height);

	convert_to_greyscale<<<grid, threads>>>(height, width, rgb_image_d, grey_image_d);
	cudaMemcpy(grey_image, grey_image_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	fprintf(file_times, "%f,", msecTotal);

	stbi_image_free(rgb_image);
	stbi_write_png("./output_GPU/0_image_grey.png", width, height, 1, grey_image, width);

	// Initialize timing

	
	
	// Apply Gaussian filtering
	
	auto kernel_size = GAUSSIAN_KERNEL_SIZE;
	float sigma = GAUSSIAN_SIGMA;
	#if LAPLACIAN_GAUSSIAN
		float* gaussian_filter = get_gaussian_laplacian_filter(kernel_size, sigma);
		#if SHARED && TILED
			cudaMemcpyToSymbol(gaussian_filter_constant, gaussian_filter, kernel_size*kernel_size*sizeof(float));
		#endif
	#else
		float* gaussian_filter = get_gaussian_filter(kernel_size, sigma);
		#if SHARED && TILED
			cudaMemcpyToSymbol(gaussian_filter_constant, gaussian_filter, kernel_size*kernel_size*sizeof(float));
		#endif
	#endif
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			std::cout<<gaussian_filter[i*kernel_size + j]<<" ";
		}
		std::cout<<std::endl;
	}

	

	uint8_t* gaussian_image;
	uint8_t* gaussian_image_d;
	float* gaussian_filter_d;
	cudaEventRecord(start, NULL);
    gaussian_image = (uint8_t*)malloc(width*height);
	cudaMalloc(&gaussian_image_d, width*height);
	cudaMalloc(&gaussian_filter_d, kernel_size*kernel_size*sizeof(float));
	cudaMemcpy(gaussian_filter_d, gaussian_filter, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);	

	#if SHARED && TILED
		auto tile_size_alt = TILE_SIZE+GAUSSIAN_KERNEL_SIZE-1;
		printf("shared and tiled\n");
		apply_filter_shared_tiled<<<grid, threads, sizeof(uint8_t)*(tile_size_alt * tile_size_alt)>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, 0);
	#else
	#if SHARED	
		printf("shared\n");
		apply_filter_shared<<<grid, threads, kernel_size*kernel_size*sizeof(float)>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, gaussian_filter_d);
	#else
		printf("global\n");
		apply_filter_global<<<grid, threads>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, gaussian_filter_d);
	#endif
	#endif

	cudaMemcpy(gaussian_image, gaussian_image_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Gaussian filtering time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_image_free(grey_image);
	stbi_write_png("./output_GPU/0_image_gaussian.png", width, height, 1, gaussian_image, width);


	// Apply 3x3 Sobel filtering


	//Apply 3x3 Sobel filtering
	float sobel_h[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
	float sobel_v[9] = {1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f};
	uint8_t* sobel_image_h;
	uint8_t* sobel_image_v;
	uint8_t* sobel_image_h_d;
	uint8_t* sobel_image_v_d;
	float* sobel_h_d;
	float* sobel_v_d;

	cudaEventRecord(start, NULL);

	sobel_image_h = (uint8_t*)malloc(width*height);
	sobel_image_v = (uint8_t*)malloc(width*height);
	cudaMalloc(&sobel_image_h_d, width*height);
	cudaMalloc(&sobel_image_v_d, width*height);
	cudaMalloc(&sobel_h_d, 9*sizeof(float));
	cudaMalloc(&sobel_v_d, 9*sizeof(float));

	cudaMemcpy(sobel_h_d, sobel_h, 9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_v_d, sobel_v, 9*sizeof(float), cudaMemcpyHostToDevice);


	#if SHARED && TILED
		apply_filter_shared_tiled<<<grid, threads, sizeof(uint8_t)*(TILE_SIZE+2)*(TILE_SIZE+2)>>>(3, height, width, sobel_image_h_d, gaussian_image_d, 1);
		apply_filter_shared_tiled<<<grid, threads, sizeof(uint8_t)*(TILE_SIZE+2)*(TILE_SIZE+2)>>>(3, height, width, sobel_image_v_d, gaussian_image_d, 2);
	#else
	#if SHARED
		apply_filter_shared<<<grid, threads, 9*sizeof(float)>>>(3, height, width, sobel_image_h_d, gaussian_image_d, sobel_h_d);
		apply_filter_shared<<<grid, threads, 9*sizeof(float)>>>(3, height, width, sobel_image_v_d, gaussian_image_d, sobel_v_d);
	#else
		apply_filter_global<<<grid, threads>>>(3, height, width, sobel_image_h_d, gaussian_image_d, sobel_h_d);
		apply_filter_global<<<grid, threads>>>(3, height, width, sobel_image_v_d, gaussian_image_d, sobel_v_d);
	#endif
	#endif

	cudaMemcpy(sobel_image_h, sobel_image_h_d, width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(sobel_image_v, sobel_image_v_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Sobel filtering time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_image_free(gaussian_image);
	stbi_write_png("./output_GPU/1_image_sobel_h.png", width, height, 1, sobel_image_h, width);
	stbi_write_png("./output_GPU/1_image_sobel_v.png", width, height, 1, sobel_image_v, width);




	// // Calculate magnitude and gradient direction

	

	float* gradient_direction;
	float* gradient_direction_d;
	uint8_t* magnitude;
	uint8_t* magnitude_d;

	cudaEventRecord(start, NULL);
	gradient_direction = (float*)malloc(width*height*sizeof(float));
	magnitude = (uint8_t*)malloc(width*height);
	cudaMalloc(&gradient_direction_d, width*height*sizeof(float));
	cudaMalloc(&magnitude_d, width*height);

	compute_magnitude_and_gradient<<<grid, threads>>>(height, width, sobel_image_h_d, sobel_image_v_d, magnitude_d, gradient_direction_d);

	cudaMemcpy(gradient_direction, gradient_direction_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(magnitude, magnitude_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Gradient/magnitude calculation time: %f (ms)\n", msecTotal);
	
	fprintf(file_times, "%f,", msecTotal);

	stbi_image_free(sobel_image_v);
	stbi_image_free(sobel_image_h);
	stbi_write_png("./output_GPU/2_gradient_direction.png", width, height, 1, gradient_direction, width);
	stbi_write_png("./output_GPU/2_magnitude.png", width, height, 1, magnitude, width);

	// // Non-maximum suppression

	cudaEventRecord(start, NULL);

	uint8_t* suppr_mag;
	uint8_t* suppr_mag_d;
	suppr_mag = (uint8_t*)malloc(width*height);
	cudaMalloc(&suppr_mag_d, width*height);
	cudaMemset(suppr_mag_d, 0, width*height);

	non_maximum_suppression_non_interpolated<<<grid, threads>>>(height, width, suppr_mag_d, magnitude_d, gradient_direction_d);

	cudaMemcpy(suppr_mag, suppr_mag_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Non-max suppression time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_image_free(magnitude);
	stbi_image_free(gradient_direction);
	stbi_write_png("./output_GPU/3_nonmax_suppr.png", width, height, 1, suppr_mag, width);

	// // float thresholding and edge tracking by hysteresis
	uint8_t* pixel_classification;
	uint8_t* pixel_classification_d;
	
	cudaEventRecord(start, NULL);
	pixel_classification = (uint8_t*)malloc(width*height);
	cudaMalloc(&pixel_classification_d, width*height*sizeof(uint8_t));



	
	float max_mag = 0.0;

	// Find the maximum and minimum magnitude values
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float mag = suppr_mag[i * width + j];
			max_mag = std::max(max_mag, mag);
			
		}
	}

	float_threshold<<<grid, threads>>>(height, width, pixel_classification_d, suppr_mag_d, max_mag);

	cudaMemcpy(pixel_classification, pixel_classification_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Double thresholding time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);
	
	stbi_write_png("./output_GPU/4_thresholded.png", width, height, 1, pixel_classification, width);
	

	// Hyteresis
	
	cudaEventRecord(start, NULL);

	#if STACK
		hysteresis_stack<<<grid, threads, 47 * 1024>>>(height, width, pixel_classification_d);
	#else
		hysteresis<<<grid, threads>>>(height, width, pixel_classification_d);
	#endif

	cudaMemcpy(pixel_classification, pixel_classification_d, width*height, cudaMemcpyDeviceToHost);

  	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Hysteresis time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_write_png("./output_GPU/5_hysteresis.png", width, height, 1, pixel_classification, width);

	// // Final timing

	


	//Dilation and Erosion

	//Dilation

	
	int dilation_kernel_size = 3;
	uint8_t* dilation;
	uint8_t* dilation_d;
		cudaEventRecord(start, NULL);

	dilation = (uint8_t*)malloc(width*height);
	cudaMalloc(&dilation_d, width*height*sizeof(uint8_t));

	//dilation kernel
	
	float dilation_kernel[dilation_kernel_size * dilation_kernel_size] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	float* dilation_kernel_d;

	
	cudaMalloc(&dilation_kernel_d, dilation_kernel_size * dilation_kernel_size * sizeof(float));
	cudaMemcpy(dilation_kernel_d, dilation_kernel, dilation_kernel_size * dilation_kernel_size * sizeof(float), cudaMemcpyHostToDevice);

	
	
	#if SHARED
		apply_dilation_shared<<<grid, threads, sizeof(float)*dilation_kernel_size*dilation_kernel_size>>>(dilation_kernel_size, height, width, dilation_d, pixel_classification_d, dilation_kernel_d);
	#else
		apply_dilation_global<<<grid, threads>>>(dilation_kernel_size, height, width, dilation_d, pixel_classification_d, dilation_kernel_d);
	#endif

	cudaMemcpy(dilation, dilation_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Dilation time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_write_png("./output_GPU/6_dilation.png", width, height, 1, dilation, width);

	// //Erosion

	uint8_t* erosion;
	uint8_t* erosion_d;
	int erosion_kernel_size = 3;	
	cudaEventRecord(start, NULL);


	//erosion kernel
	erosion = (uint8_t*)malloc(width*height);

	float erosion_kernel[erosion_kernel_size*erosion_kernel_size] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	float* erosion_kernel_d;
	
	cudaMalloc(&erosion_d, width*height);
	cudaMalloc(&erosion_kernel_d, erosion_kernel_size*erosion_kernel_size*sizeof(float));

	cudaMemcpy(erosion_kernel_d, erosion_kernel, erosion_kernel_size*erosion_kernel_size*sizeof(float), cudaMemcpyHostToDevice);

	#if SHARED
		apply_erosion_shared<<<grid, threads, sizeof(float)*erosion_kernel_size*erosion_kernel_size>>>(erosion_kernel_size, height, width, erosion_d, dilation_d, erosion_kernel_d);
	#else
		apply_erosion_global<<<grid, threads>>>(erosion_kernel_size, height, width, erosion_d, dilation_d, erosion_kernel_d);
	#endif
	cudaMemcpy(erosion, erosion_d, width*height, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Erosion time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);
	stbi_write_png("./output_GPU/7_erosion.png", width, height, 1, erosion, width);

	//HOUGH TRANSFORM


	int max_rho = (int)sqrt((width*width) + (height*height));
	int max_theta = 180;
	int* hough_space;
	int* hough_space_d;
	auto channels = 3;
	uint8_t* hough_output;
	uint8_t* hough_output_d;
	int* lines_d;

	cudaEventRecord(start, NULL);
    hough_space = new int[max_rho * max_theta](); // Initialize to 0


	// for (int i=0 ; i< max_rho; i++){
	// 	for (int j=0 ; j< max_theta; j++){
	// 		printf("%d ", hough_space[i*max_rho + j]);
	// 	}
	// } 
	
	cudaMalloc(&hough_space_d, max_rho*max_theta*sizeof(int));
	cudaMemset(hough_space_d, 0, max_rho*max_theta*sizeof(int));
	hough_output = (uint8_t*)malloc(width*height*channels);
	hough_output_d = rgb_image_d;

	cudaMalloc(&lines_d, 2*sizeof(int)*max_rho*max_theta);
	cudaMemset(lines_d, 0, 2*sizeof(int)*max_rho*max_theta);
	
	hough_transform<<<grid, threads>>>(height, width, max_rho, max_theta, HOUGH_THRESHOLD_MULT, erosion_d, hough_space_d, hough_output_d, channels, lines_d);

	cudaMemcpy(hough_space, hough_space_d, max_rho*max_theta*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hough_output, hough_output_d, width*height*channels, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Hough transform time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f,", msecTotal);

	stbi_write_png("./output_GPU/8_hough_space.png", max_theta, max_rho, 1, hough_space, max_theta);
	stbi_write_png("./output_GPU/8_hough_output.png", width, height, channels, hough_output, width*channels);

	cudaEventRecord(stop_total, NULL);
  	cudaEventSynchronize(stop_total);
  	cudaEventElapsedTime(&msecTotal, start_total, stop_total);

	printf("\t Total time: %f (ms)\n", msecTotal);
	fprintf(file_times, "%f\n", msecTotal);



    return 0;
}