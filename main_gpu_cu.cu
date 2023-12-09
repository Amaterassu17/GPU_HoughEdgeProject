#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


__global__ void apply_filter(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
	// for(int i = 1; i < height-1; i++)
 	// {
    //  	for(int j = 1; j < width-1; j++)
	//  	{
	// 		float sum = 0;

	// 		for(int k = 0; k < kernel_size; k++)
	//  		{
	// 			for(int m = 0; m < kernel_size; m++)
	//  			{
	// 				sum += kernel[k*kernel_size + m]*input[(i+(k-1))*width + j + (m-1)];

	// 			}
	// 		}
	// 		output[i*width + j] = (int)abs(sum);
    //  	}
 	// }

    //TODO: PARALLELIZE

}

__global__ void convert_to_greyscale(int height, int width, uint8_t *img, uint8_t *grey_img)
{
	// for(int i = 0; i < height; i++)
 	// {
    //  	for(int j = 0; j < width; j++)
	//  	{
	// 		auto b = img[i*width*3 + j*3 + 0];
	// 		auto g = img[i*width*3 + j*3 + 1];
	// 		auto r = img[i*width*3 + j*3 + 2];

	// 		int average = (int)(0.2126*r+0.7152*g+0.0722*b);

	// 		grey_img[i*width + j] = average;
    //  	}
 	// }
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



int main(int argc, char *argv[])
{
    //Cuda definitions
    const int blocksize = 16;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int device;
    struct cudaDeviceProp properties;
    

    cudaError_t err = cudaSuccess;
    cudaDeviceProp deviceProp;
    int devID = 0;
    error = cudaGetDevice(&devID);

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

    threads = dim3(blocksize, blocksize);
    grid = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    printf("CUDA kernel launch with %d blocks of %d threads\n", grid.x * grid.y, threads.x * threads.y);

    //image definitions
    int width, height, bpp;
    
	auto img_fname = argc>=2 ? argv[1] : "image.png";

	system("mkdir -p output");
	auto file_times = fopen("./output/times.txt", "w");
	

    //program starts

	uint8_t* rgb_image = stbi_load(img_fname, &width, &height, &bpp, 3);
    uint8_t* rgb_image_d;
    cudaMalloc(&rgb_image_d, width*height*3);
    cudaMemcpy(rgb_image_d, rgb_image, width*height*3, cudaMemcpyHostToDevice);


    std::cout<<"image: "<<img_fname<<std::endl;
	std::cout<<width<<" "<<height<<std::endl;

    //Stop here

	// Convert to greyscale
    uint8_t* grey_image;
    grey_image = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "convert_to_greyscale");
	convert_to_greyscale(height, width, rgb_image, grey_image);
	measure_time(false, file_times, "convert_to_greyscale");
	stbi_image_free(rgb_image);

	// Apply Gaussian filtering
	int kernel_size = 3;
	float gaussian_filter[kernel_size*kernel_size] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
	uint8_t* gaussian_image;
    gaussian_image = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "apply_gaussian_filter");
	apply_filter(kernel_size, height, width, gaussian_image, grey_image, gaussian_filter);
	measure_time(false, file_times, "apply_gaussian_filter");

	stbi_image_free(grey_image);
	stbi_write_png("./output/0_image_gaussian.png", width, height, 1, gaussian_image, width);
	
	// Apply Sobel filtering
	float sobel_h[kernel_size*kernel_size] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float sobel_v[kernel_size*kernel_size] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	uint8_t* sobel_image_h;
    sobel_image_h = (uint8_t*)malloc(width*height);
	uint8_t* sobel_image_v;
    sobel_image_v = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "apply_sobel_filters");
	apply_filter(kernel_size, height, width, sobel_image_h, gaussian_image, sobel_h);
	apply_filter(kernel_size, height, width, sobel_image_v, gaussian_image, sobel_v);
	measure_time(false, file_times, "apply_sobel_filters");

    stbi_image_free(gaussian_image);
	stbi_write_png("./output/1_image_sobel_h.png", width, height, 1, sobel_image_h, width);
	stbi_write_png("./output/1_image_sobel_v.png", width, height, 1, sobel_image_v, width);

	// Calculate magnitude and gradient direction
    float* gradient_direction;
    gradient_direction = (float*)malloc(width*height*sizeof(float));
	uint8_t* magnitude;
    magnitude = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "compute_magnitude_and_gradient");
	compute_magnitude_and_gradient(height, width, sobel_image_h, sobel_image_v, magnitude, gradient_direction);
	measure_time(false, file_times, "compute_magnitude_and_gradient");

	stbi_image_free(sobel_image_v);
	stbi_image_free(sobel_image_h);
	stbi_write_png("./output/2_gradient_direction.png", width, height, 1, gradient_direction, width);
	stbi_write_png("./output/2_magnitude.png", width, height, 1, magnitude, width);

	// Non-maximum suppression
	uint8_t* suppr_mag;
    suppr_mag = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "non_maximum_suppression");
	non_maximum_suppression(height, width, suppr_mag, magnitude, gradient_direction);
	measure_time(false, file_times, "non_maximum_suppression");

	double max = *std::max_element(magnitude, magnitude + width*height);

	std::cout<<max<<std::endl;

	stbi_image_free(magnitude);
	stbi_image_free(gradient_direction);
	stbi_write_png("./output/3_nonmax_suppr.png", width, height, 1, suppr_mag, width);

	// classify pixels as strong, weak or non-relevant
	uint8_t* pixel_classification;
    pixel_classification = (uint8_t*)malloc(width*height);
	
	measure_time(true, file_times, "double_threshold");
	double_threshold(height, width, pixel_classification, suppr_mag);
	measure_time(false, file_times, "double_threshold");

	stbi_write_png("./output/4_thresholded.png", width, height, 1, pixel_classification, width);

	measure_time(true, file_times, "hysteresis");
	hysteresis(height, width, pixel_classification);
	measure_time(false, file_times, "hysteresis");

	stbi_write_png("./output/5_hysteresis.png", width, height, 1, pixel_classification, width);

    return 0;
}


// TODOS:
// zeit messen