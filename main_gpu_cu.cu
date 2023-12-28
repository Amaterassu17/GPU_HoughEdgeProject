#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 16


//Should be the threshold based on the max magnitudes seen in the image. In our case most likely 255

#define LAPLACIAN_GAUSSIAN 0
#define GAUSSIAN_KERNEL_SIZE 3
#define GAUSSIAN_SIGMA 1.1




#define MAX_THRESHOLD_MULT 0.15
#define MIN_THRESHOLD_MULT 0.02
#define NON_MAX_SUPPR_THRESHOLD 0.4



__global__ void apply_filter_global(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	// Print the matrix kernel
	//  if (i == 1 && j == 1 ){
	//  	printf("Kernel: \n");
	// 	for (int p = 0; p < kernel_size; p++) {
	// 		for (int o = 0; o < kernel_size; o++) {
	// 			printf("%f ", kernel[p * kernel_size + o]);
	// 		}
	// 	}
	// 	printf("\n");
	//  }

	//print input image 
	//printf("%u ", input[i * width + j]);

	float sum = 0;

	
	if(i < height && j < width){
		for (int k = 0; k < kernel_size; k++)
		{
			for (int m = 0; m < kernel_size; m++)
			{
				sum += kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)];
				//sum +=  input[i * width + j];
			}
		}
	}

	//printf("sum: %f\n", sum);
	output[i * width + j] = (int)abs(sum);
	if(output[i * width + j] != 0)
		printf("output: %u\n", output[i * width + j]);
}

__global__ void apply_filter_shared(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	extern __shared__ float kernel_shared[];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	// for(int p = 0; p < kernel_size; p++){
	// 	for(int o = 0; o < kernel_size; o++){
	// 		printf("%f ", kernel[p*kernel_size + o]);
	// 	}
	// 	printf("\n");
	// }


	if(blockIdx.x < kernel_size && blockIdx.y < kernel_size){
		kernel_shared[threadIdx.y*kernel_size + threadIdx.x] = kernel[threadIdx.y*kernel_size + threadIdx.x];
	}

	if(i == 0 && j==0)
	for(int p = 0; p < kernel_size; p++){
		for(int o = 0; o < kernel_size; o++){
			printf("%f ", kernel_shared[p*kernel_size + o]);
		}
		printf("\n");
	}
	

	__syncthreads();


	if(i < height && j < width){
		float sum = 0;

		for (int k = 0; k < kernel_size; k++)
		{
			for (int m = 0; m < kernel_size; m++)
			{
				sum += kernel_shared[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)];
				//sum +=  input[i * width + j];
			}
		}
		output[i * width + j] = (int)abs(sum);
	}

}

__global__ void apply_filter_shared_tiled(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	
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

__global__ void compute_magnitude_and_gradient(int height, int width, uint8_t *Ix, uint8_t *Iy, uint8_t *mag, float *grad){

	// for(int i = 1; i < height-1; i++)
 	// {
    //  	for(int j = 1; j < width-1; j++)
	//  	{
	// 		float dx = Ix[i*width+j];
	// 		float dy = Iy[i*width+j];
	// 		mag[i*width+j] = (int)sqrt(dx*dx+dy*dy);
	// 		float angle = atan2(dy, dx)*180/M_PI;
	// 		grad[i*width+j] = angle < 180 ? angle+180 : angle;
    //  	}
 	// }

	

}

__global__ void non_maximum_suppression(int height, int width, uint8_t *suppr_mag, uint8_t *mag, float* grad){

	// for(int i = 0; i < height; i++)
 	// {
    //  	for(int j = 0; j < width; j++)
	//  	{
	// 		// in cpp is there a better way to initialize with zeros??
	// 		suppr_mag[i*width + j] = 0;
    //  	}
 	// }


	// for(int i = 1; i < height-1; i++)
 	// {
    //  	for(int j = 1; j < width-1; j++)
	//  	{
	// 		int q = 255;
	// 		int r = 255;

	// 		//angle 0
    //         if (0 <= grad[i*width+j] < 22.5 || 157.5 <= grad[i*width+j] <= 180){
    //             q = mag[i*width + j+1];
    //             r = mag[i*width + j-1];
	// 		}
    //         //angle 45
    //         else if (22.5 <= grad[i*width+j] < 67.5){
    //             q = mag[(i+1)*width + j-1];
    //             r = mag[(i-1)*width + j+1];
	// 		}
    //         //angle 90
    //         else if (67.5 <= grad[i*width+j] < 112.5){
    //             q = mag[(i+1)*width + j];
    //             r = mag[(i-1)*width + j];
	// 		}
    //         //angle 135
    //         else if (112.5 <= grad[i*width+j] < 157.5){
    //             q = mag[(i-1)*width + j-1];
    //             r = mag[(i+1)*width + j+1];
	// 		}

	// 		if (mag[i*width + j] >= q && mag[i*width + j] >= r){
    //             suppr_mag[i*width + j] = mag[i*width + j];
	// 		} else {
	// 			suppr_mag[i*width + j] = 0;
	// 		}

    //  	}
 	// }



}

__global__ void double_threshold(int height, int width,  uint8_t *pixel_classification,  uint8_t *suppr_mag){

	// float high_threshold = 0.09*255;
	// float low_threshold = high_threshold*0.05;

	// std::cout<<low_threshold<<", "<<high_threshold<<std::endl;
	
	// for(int i = 0; i < height; i++)
 	// {
    //  	for(int j = 0; j < width; j++)
	//  	{
	// 		if(suppr_mag[i*width+j] >= high_threshold){
	// 			// strong pixels
	// 			pixel_classification[i*width+j] = 255;
	// 		} else if (suppr_mag[i*width+j] < low_threshold){
	// 			// non relevant pixels
	// 			pixel_classification[i*width+j] = 0;
	// 		} else {
	// 			// weak pixels
	// 			pixel_classification[i*width+j] = 25;
	// 		}
    //  	}
 	// }



}

__global__ void hysteresis(int height, int width, uint8_t *pixel_classification){

	// for(int i = 1; i < height-1; i++)
 	// {
    //  	for(int j = 1; j < width-1; j++)
	//  	{
	// 		if(pixel_classification[i*width+j] == 25){
	// 			if(pixel_classification[(i+1)*width+j-1] == 255 || pixel_classification[(i+1)*width+j] == 255 || pixel_classification[(i+1)*width+j+1] == 255 ||
	// 			pixel_classification[i*width+j-1] == 255 || pixel_classification[i*width+j+1] == 255 || pixel_classification[(i-1)*width+j-1] == 255 ||
	// 			pixel_classification[(i-1)*width+j] == 255 || pixel_classification[(i-1)*width+j+1] == 255){
	// 				pixel_classification[i*width + j] = 255;
	// 			} else {
	// 				pixel_classification[i*width + j] = 0;
	// 			}
	// 		}
    //  	}
 	// }
    


}


// void measure_time(bool start, FILE* file_times, std::string name){
// 	static std::chrono::system_clock::time_point start_time;
// 	static std::chrono::system_clock::time_point end_time;
// 	if(start){
// 		start_time = std::chrono::system_clock::now();
// 	} else {
// 		end_time = std::chrono::system_clock::now();
// 		std::chrono::duration<double> duration = end_time - start_time;
// 		fprintf(file_times, "%s: %f \n", name.c_str(), duration.count());
// 	}
// }

float* get_gaussian_filter (int kernel_size, float sigma){

	kernel_size = kernel_size%2 == 0 ? kernel_size-1 : kernel_size;

	float* gaussian_filter = (float*)malloc(kernel_size*kernel_size*sizeof(float));
	float sum = 0.0;
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			gaussian_filter[i*kernel_size + j] = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
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
    //Cuda definitions
    const int blocksize = BLOCK_SIZE;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
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
	auto file_times = fopen("./output/times.txt", "w");
	

    //program starts

	uint8_t* rgb_image = stbi_load(img_fname, &width, &height, &bpp, 3);
    uint8_t* rgb_image_d;
    cudaMalloc(&rgb_image_d, width*height*3);
    cudaMemcpy(rgb_image_d, rgb_image, width*height*3, cudaMemcpyHostToDevice);

	threads = dim3(blocksize, blocksize);
    grid = dim3((width + threads.x - 1) / threads.x , (height + threads.y - 1) / threads.y);
    printf("CUDA kernel launch with %d blocks of %d threads\n", grid.x * grid.y, threads.x * threads.y);



    std::cout<<"image: "<<img_fname<<std::endl;
	std::cout<<width<<" "<<height<<std::endl;

    //Stop here

	// Convert to greyscale
	uint8_t* grey_image;
	uint8_t* grey_image_d;
	grey_image = (uint8_t*)malloc(width*height);
	cudaMalloc(&grey_image_d, width*height);

	//measure_time(true, file_times, "convert_to_greyscale");
	// convert_to_greyscale(height, width, rgb_image, grey_image);
	convert_to_greyscale<<<grid, threads>>>(height, width, rgb_image_d, grey_image_d);
	cudaDeviceSynchronize(); // Synchronize with CUDA
	cudaMemcpy(grey_image, grey_image_d, width*height, cudaMemcpyDeviceToHost);

	//print
	// for(int i = 0; i < height; i++){
	// 	for(int j = 0; j < width; j++){
	// 		printf("%d ", grey_image[i*width + j]);
	// 	}
	// 	printf("\n");
	// }

	//measure_time(false, file_times, "convert_to_greyscale");
	stbi_image_free(rgb_image);
	cudaFree(rgb_image_d);
	stbi_write_png("./output_GPU/0_image_grey.png", width, height, 1, grey_image, width);

	


	// Apply Gaussian filtering
	
	auto kernel_size = GAUSSIAN_KERNEL_SIZE;
	float sigma = GAUSSIAN_SIGMA;
	#if LAPLACIAN_GAUSSIAN
		float* gaussian_filter = get_gaussian_laplacian_filter(kernel_size, sigma);
	#else
		float* gaussian_filter = get_gaussian_filter(kernel_size, sigma);
	#endif
	//float gaussian_filter[9] = {1,1,1,1,1,1,1,1,1};
	for(int i = 0; i < kernel_size; i++){
		for(int j = 0; j < kernel_size; j++){
			std::cout<<gaussian_filter[i*kernel_size + j]<<" ";
		}
		std::cout<<std::endl;
	}

	uint8_t* gaussian_image;
	uint8_t* gaussian_image_d;
	float* gaussian_filter_d;
    gaussian_image = (uint8_t*)malloc(width*height);
	cudaMalloc(&gaussian_image_d, width*height);
	cudaMalloc(&gaussian_filter_d, kernel_size*kernel_size*sizeof(float));
	printf("Copying input data from the host memory to the CUDA device\n");
	cudaMemcpy(gaussian_filter_d, gaussian_filter, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);	
	printf("copied");
	// static global memory version
	apply_filter_global<<<grid, threads>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, gaussian_filter_d);

	//dinamic shared memory version
	// apply_filter_shared<<<grid, threads, kernel_size*kernel_size*sizeof(float)>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, gaussian_filter_d);


	cudaMemcpy(gaussian_image, gaussian_image_d, width*height, cudaMemcpyDeviceToHost);
	//print image 
	// for(int i = 0; i < height; i++){
	// 	for(int j = 0; j < width; j++){
	// 		printf("%u ", gaussian_image[i*width + j]);
	// 	}
	// 	printf("\n");
	// }

	//free(gaussian_filter);
	stbi_image_free(grey_image);
	stbi_write_png("./output_GPU/0_image_gaussian.png", width, height, 1, gaussian_image, width);


	// measure_time(true, file_times, "apply_gaussian_filter");
	// apply_filter(kernel_size, height, width, gaussian_image, grey_image, gaussian_filter);
	// measure_time(false, file_times, "apply_gaussian_filter");

	// stbi_image_free(grey_image);
	// stbi_write_png("./output/0_image_gaussian.png", width, height, 1, gaussian_image, width);
	
	// // Apply Sobel filtering
	// float sobel_h[kernel_size*kernel_size] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	// float sobel_v[kernel_size*kernel_size] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	// uint8_t* sobel_image_h;
    // sobel_image_h = (uint8_t*)malloc(width*height);
	// uint8_t* sobel_image_v;
    // sobel_image_v = (uint8_t*)malloc(width*height);

	// measure_time(true, file_times, "apply_sobel_filters");
	// apply_filter(kernel_size, height, width, sobel_image_h, gaussian_image, sobel_h);
	// apply_filter(kernel_size, height, width, sobel_image_v, gaussian_image, sobel_v);
	// measure_time(false, file_times, "apply_sobel_filters");

    // stbi_image_free(gaussian_image);
	// stbi_write_png("./output/1_image_sobel_h.png", width, height, 1, sobel_image_h, width);
	// stbi_write_png("./output/1_image_sobel_v.png", width, height, 1, sobel_image_v, width);

	// // Calculate magnitude and gradient direction
    // float* gradient_direction;
    // gradient_direction = (float*)malloc(width*height*sizeof(float));
	// uint8_t* magnitude;
    // magnitude = (uint8_t*)malloc(width*height);

	// measure_time(true, file_times, "compute_magnitude_and_gradient");
	// compute_magnitude_and_gradient(height, width, sobel_image_h, sobel_image_v, magnitude, gradient_direction);
	// measure_time(false, file_times, "compute_magnitude_and_gradient");

	// stbi_image_free(sobel_image_v);
	// stbi_image_free(sobel_image_h);
	// stbi_write_png("./output/2_gradient_direction.png", width, height, 1, gradient_direction, width);
	// stbi_write_png("./output/2_magnitude.png", width, height, 1, magnitude, width);

	// // Non-maximum suppression
	// uint8_t* suppr_mag;
    // suppr_mag = (uint8_t*)malloc(width*height);

	// measure_time(true, file_times, "non_maximum_suppression");
	// non_maximum_suppression(height, width, suppr_mag, magnitude, gradient_direction);
	// measure_time(false, file_times, "non_maximum_suppression");

	// double max = *std::max_element(magnitude, magnitude + width*height);

	// std::cout<<max<<std::endl;

	// stbi_image_free(magnitude);
	// stbi_image_free(gradient_direction);
	// stbi_write_png("./output/3_nonmax_suppr.png", width, height, 1, suppr_mag, width);

	// // classify pixels as strong, weak or non-relevant
	// uint8_t* pixel_classification;
    // pixel_classification = (uint8_t*)malloc(width*height);
	
	// measure_time(true, file_times, "double_threshold");
	// double_threshold(height, width, pixel_classification, suppr_mag);
	// measure_time(false, file_times, "double_threshold");

	// stbi_write_png("./output/4_thresholded.png", width, height, 1, pixel_classification, width);

	// measure_time(true, file_times, "hysteresis");
	// hysteresis(height, width, pixel_classification);
	// measure_time(false, file_times, "hysteresis");

	// stbi_write_png("./output/5_hysteresis.png", width, height, 1, pixel_classification, width);

    return 0;
}


// TODOS:
// zeit messen