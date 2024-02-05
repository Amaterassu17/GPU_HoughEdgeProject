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
#define TILE_SIZE 16


#define LAPLACIAN_GAUSSIAN 0
#define GAUSSIAN_KERNEL_SIZE 3
#define GAUSSIAN_SIGMA 1.0
#define SHARED 1
#define TILED 0


#define MAX_THRESHOLD_MULT 0.15
#define MIN_THRESHOLD_MULT 0.02
#define NON_MAX_SUPPR_THRESHOLD 0.4



__global__ void apply_filter_global(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	
	// for filtering it would also make sense to do both gaussian and sobel filter one after the other
	// to avoid writing the output of gaussian to global
	// maybe also the greyscaling
	
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

		//printf("%f\n", sum);
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

		// printf("%d, ", input[i*width + j]);
		//printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", kernel_shared[0], kernel_shared[1], kernel_shared[2], kernel_shared[3], kernel_shared[4], kernel_shared[5], kernel_shared[6], kernel_shared[7], kernel_shared[8]);
		for (int k = 0; k < kernel_size; k++) {
			for (int m = 0; m < kernel_size; m++) {
				int input_row = i + (k - 1);
				int input_col = j + (m - 1);

				// Check if the indices are within bounds
				if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
					sum += kernel_shared[k * kernel_size + m] * input[input_row * width + input_col];
					if(threadIdx.x == 1 && threadIdx.y == 1){
						//printf("k = %d, m= %d, kernel_shared -> %f, input -> %d\n",k,m, kernel_shared[k * kernel_size + m], input[input_row * width + input_col]);
					}
				}
			}
		}	

		output[i * width + j] = abs(sum);
	}
}


__global__ void apply_filter_shared_tiled(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel){
	
	extern __shared__ float kernel_shared[];
	__shared__ uint8_t input_shared[TILE_SIZE][TILE_SIZE];

	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
		kernel_shared[threadIdx.y * kernel_size + threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
		input_shared[threadIdx.y][threadIdx.x] = input[i*width + j];
		//printf("loaded to shared memory");
	}

	if (threadIdx.x == 0 && threadIdx.y == 0){
		printf("i = %d, j = %d, input_shared -> %d\n", i, j, input_shared[threadIdx.y][threadIdx.x]);
	}


	__syncthreads(); // Ensure all threads have finished copying to shared memory


	if (i < height && j < width) {
		float sum = 0;

		// printf("%d, ", input[i*width + j]);
		//printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", kernel_shared[0], kernel_shared[1], kernel_shared[2], kernel_shared[3], kernel_shared[4], kernel_shared[5], kernel_shared[6], kernel_shared[7], kernel_shared[8]);
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

__global__ void compute_magnitude_and_gradient(int height, int width, uint8_t *Ix, uint8_t *Iy, uint8_t *mag, float *grad){

	// can we use shared memory for this??
	// I think not necessary because each thread accesses the corresponding element
	// no two threads access the same element therefore shared memory doesn't save any memory loads

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

__global__ void non_maximum_suppression(int height, int width, uint8_t *suppr_mag, uint8_t *mag, float* grad){

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

		if (mag_ij >= q && mag_ij >= r){
			suppr_mag[i*width + j] = mag_ij;
		} else {
			suppr_mag[i*width + j] = 0;
		}
	}


}

__global__ void double_threshold(int height, int width,  uint8_t *pixel_classification,  uint8_t *suppr_mag){

	// float high_threshold = 0.09*255;
	// float low_threshold = high_threshold*0.05;

	float high_threshold = 0.4*255;
	float low_threshold = 0.1*255;

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


__global__ void hysteresis(int height, int width, uint8_t *pixel_classification){

	// TODO: implement like in paper
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

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
			for(int m=-1; m<1; m++){
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
	// max 8 * width*height ints on the stack if image is completely white
	// that's 8 * 1024 * 1024 = 8* 10^6 ints
	// does it fit in shared memory?
	// one int is 4B
	// = 32 * 10^6 B = 32 000 kB (but we only have 48kB)
	// which means a stack depth of 12 000
	// ahhh but we are tiling so it should be fine
	__syncthreads(); // is this correct here? now all neighbouring weak edge pixels should be pushed
	// though not necessarily in the right order but i think it doesn't matter here
	
	while(stack_depth > 0){
		// tmp_stack_depth stores value before 1 was subtracted
		int tmp_stack_depth = atomicSub(&stack_depth, 1);

		if(tmp_stack_depth > 0){
			// every stack pops a different element
			int index = stack_sdata[tmp_stack_depth-1];
			// set current one to final edge pixel
			pixel_classification[index] = 255;
			int i_top = index % width;
			int j_top = index - i_top*width;
			for(int l=-1; l<2; l++){
				for(int m=-1; m<1; m++){
					int new_idx = (i_top+l)*width+(j_top+m);
					if(m != 0 && l !=0 && (i_top+l) > 0 && (j_top+m) > 0 && (i_top+l) < height & (j_top+m) < width &&
					pixel_classification[new_idx] == 50){
						// neighbour is valid weak edge pixel -> push to stack
						int tmp_stack_depth = atomicAdd(&stack_depth, 1);
						stack_sdata[tmp_stack_depth] = new_idx;
					}
				}
			}
		} else {
			// subtracted one to many, add one
			atomicAdd(&stack_depth, 1);
		}
	}

	__syncthreads();

	// remove remaining weak edge pixels
	if(i < height && j < width && pixel_classification[i*width+j] == 50){
		//printf("%d removing weak pixels \n", 2);
		pixel_classification[i*width+j] = 0;
	} else{
		//printf("%d %d, %d why not executing \n", i, j, pixel_classification[i*width+j]);
	}

	__syncthreads();
}


__global__ void apply_dilation(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

// if(i < height && j < width){
// 	float max_val = std::numeric_limits<float>::min();

// 	for (int k = 0; k < kernel_size; k++) {
// 		for (int m = 0; m < kernel_size; m++) {
// 			float pixel_value = kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)];
// 			max_val = std::max(max_val, pixel_value);
// 		}
// 	}

// 	output[i * width + j] = max_val;
// }


}

__global__ void apply_erosion(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

// if(i < height && j < width){
// 	float min_val = std::numeric_limits<float>::max();

// 	for (int k = 0; k < kernel_size; k++) {
// 		for (int m = 0; m < kernel_size; m++) {
// 			min_val = std::min(min_val, kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)]);
// 		}
// 	}
// 	output[i * width + j] = (int)min_val;

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
	cudaMemcpy(grey_image, grey_image_d, width*height, cudaMemcpyDeviceToHost);

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
	cudaMemcpy(gaussian_filter_d, gaussian_filter, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);	

	#if SHARED && TILED
		printf("shared and tiled\n");
		apply_filter_shared_tiled<<<grid, threads, kernel_size*kernel_size*sizeof(float) + sizeof(uint8_t)*(TILE_SIZE * TILE_SIZE)>>>(kernel_size, height, width, gaussian_image_d, grey_image_d, gaussian_filter_d);
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
	stbi_image_free(grey_image);
	stbi_write_png("./output_GPU/0_image_gaussian.png", width, height, 1, gaussian_image, width);

	

	//Apply 3x3 Sobel filtering
	float sobel_h[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float sobel_v[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	uint8_t* sobel_image_h;
	uint8_t* sobel_image_v;
	uint8_t* sobel_image_h_d;
	uint8_t* sobel_image_v_d;
	float* sobel_h_d;
	float* sobel_v_d;

	sobel_image_h = (uint8_t*)malloc(width*height);
	sobel_image_v = (uint8_t*)malloc(width*height);
	cudaMalloc(&sobel_image_h_d, width*height);
	cudaMalloc(&sobel_image_v_d, width*height);
	cudaMalloc(&sobel_h_d, 9*sizeof(float));
	cudaMalloc(&sobel_v_d, 9*sizeof(float));

	cudaMemcpy(sobel_h_d, sobel_h, 9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_v_d, sobel_v, 9*sizeof(float), cudaMemcpyHostToDevice);

	#if SHARED
		apply_filter_shared<<<grid, threads, 9*sizeof(float)>>>(3, height, width, sobel_image_h_d, gaussian_image_d, sobel_h_d);
		apply_filter_shared<<<grid, threads, 9*sizeof(float)>>>(3, height, width, sobel_image_v_d, gaussian_image_d, sobel_v_d);
	#else
		apply_filter_global<<<grid, threads>>>(3, height, width, sobel_image_h_d, gaussian_image_d, sobel_h_d);
		apply_filter_global<<<grid, threads>>>(3, height, width, sobel_image_v_d, gaussian_image_d, sobel_v_d);
	#endif

	cudaMemcpy(sobel_image_h, sobel_image_h_d, width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(sobel_image_v, sobel_image_v_d, width*height, cudaMemcpyDeviceToHost);

	stbi_image_free(gaussian_image);
	stbi_write_png("./output_GPU/1_image_sobel_h.png", width, height, 1, sobel_image_h, width);
	stbi_write_png("./output_GPU/1_image_sobel_v.png", width, height, 1, sobel_image_v, width);




	// // Calculate magnitude and gradient direction
    

	float* gradient_direction;
	float* gradient_direction_d;
	uint8_t* magnitude;
	uint8_t* magnitude_d;
	
	gradient_direction = (float*)malloc(width*height*sizeof(float));
	magnitude = (uint8_t*)malloc(width*height);
	cudaMalloc(&gradient_direction_d, width*height*sizeof(float));
	cudaMalloc(&magnitude_d, width*height);

	compute_magnitude_and_gradient<<<grid, threads>>>(height, width, sobel_image_h_d, sobel_image_v_d, magnitude_d, gradient_direction_d);

	cudaMemcpy(gradient_direction, gradient_direction_d, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(magnitude, magnitude_d, width*height, cudaMemcpyDeviceToHost);

	stbi_image_free(sobel_image_v);
	stbi_image_free(sobel_image_h);
	stbi_write_png("./output_GPU/2_gradient_direction.png", width, height, 1, gradient_direction, width);
	stbi_write_png("./output_GPU/2_magnitude.png", width, height, 1, magnitude, width);

	// // Non-maximum suppression
	uint8_t* suppr_mag;
	uint8_t* suppr_mag_d;
	suppr_mag = (uint8_t*)malloc(width*height);
	cudaMalloc(&suppr_mag_d, width*height);

	non_maximum_suppression<<<grid, threads>>>(height, width, suppr_mag_d, magnitude_d, gradient_direction_d);

	cudaMemcpy(suppr_mag, suppr_mag_d, width*height, cudaMemcpyDeviceToHost);

	stbi_image_free(magnitude);
	stbi_image_free(gradient_direction);
	stbi_write_png("./output_GPU/3_nonmax_suppr.png", width, height, 1, suppr_mag, width);

	// // Double thresholding and edge tracking by hysteresis
	uint8_t* pixel_classification;
	uint8_t* pixel_classification_d;
	pixel_classification = (uint8_t*)malloc(width*height);
	cudaMalloc(&pixel_classification_d, width*height);

	double_threshold<<<grid, threads>>>(height, width, pixel_classification_d, suppr_mag_d);

	cudaMemcpy(pixel_classification, pixel_classification_d, width*height, cudaMemcpyDeviceToHost);
	
	stbi_write_png("./output_GPU/4_thresholded.png", width, height, 1, pixel_classification, width);
	
	cudaEvent_t start, stop;
  	float msecTotal;
    cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	//hysteresis_stack<<<grid, threads, 47 * 1024>>>(height, width, pixel_classification_d);
	hysteresis<<<grid, threads>>>(height, width, pixel_classification_d);
	cudaEventCreate(&stop);
  	cudaEventRecord(stop, NULL);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&msecTotal, start, stop);

	printf("\t Processing time: %f (ms)", msecTotal);

	cudaMemcpy(pixel_classification, pixel_classification_d, width*height, cudaMemcpyDeviceToHost);

	stbi_write_png("./output_GPU/5_hysteresis.png", width, height, 1, pixel_classification, width);


	//Dilation and Erosion

	//Dilation

	uint8_t* dilation;
	uint8_t* dilation_d;
	dilation = (uint8_t*)malloc(width*height);
	cudaMalloc(&dilation_d, width*height);

	//dilation kernel
	auto dilation_kernel_size = 3;
	float dilation_kernel[dilation_kernel_size*dilation_kernel_size] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	float* dilation_kernel_d;
	cudaMalloc(&dilation_kernel_d, dilation_kernel_size*dilation_kernel_size*sizeof(float));
	cudaMemcpy(dilation_kernel_d, dilation_kernel, dilation_kernel_size*dilation_kernel_size*sizeof(float), cudaMemcpyHostToDevice);

	apply_dilation<<<grid, threads>>>(dilation_kernel_size, height, width, dilation_d, pixel_classification_d, dilation_kernel);

	cudaMemcpy(dilation, dilation_d, width*height, cudaMemcpyDeviceToHost);

	stbi_write_png("./output_GPU/6_dilation.png", width, height, 1, dilation, width);

	// //Erosion

	uint8_t* erosion;
	uint8_t* erosion_d;
	erosion = (uint8_t*)malloc(width*height);
	cudaMalloc(&erosion_d, width*height);

	//erosion kernel
	auto erosion_kernel_size = 3;
	float erosion_kernel[erosion_kernel_size*erosion_kernel_size] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	float* erosion_kernel_d;
	cudaMalloc(&erosion_kernel_d, erosion_kernel_size*erosion_kernel_size*sizeof(float));

	cudaMemcpy(erosion_kernel_d, erosion_kernel, erosion_kernel_size*erosion_kernel_size*sizeof(float), cudaMemcpyHostToDevice);

	apply_erosion<<<grid, threads>>>(erosion_kernel_size, height, width, erosion_d, dilation_d, erosion_kernel);

	cudaMemcpy(erosion, erosion_d, width*height, cudaMemcpyDeviceToHost);

	stbi_write_png("./output_GPU/7_erosion.png", width, height, 1, erosion, width);
    return 0;
}