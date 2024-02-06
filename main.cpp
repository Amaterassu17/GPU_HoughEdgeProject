// Edge detection

#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>



#define LAPLACIAN_GAUSSIAN 0
#define GAUSSIAN_KERNEL_SIZE 5
#define GAUSSIAN_SIGMA 1.0

#define MAX_THRESHOLD_MULT 0.15
#define MIN_THRESHOLD_MULT 0.02
#define NON_MAX_SUPPR_THRESHOLD 0.4


void apply_filter(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
	for(int i = 1; i < height-1; i++)
 	{
     	for(int j = 1; j < width-1; j++)
	 	{
			float sum = 0;

			for(int k = 0; k < kernel_size; k++)
	 		{
				for(int m = 0; m < kernel_size; m++)
	 			{
					int input_row = i+(k-1);
					int input_col = j+(m-1);
					if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
						sum += kernel[k * kernel_size + m]*input[input_row * width + input_col];
					}
				}
			}
			output[i*width + j] = abs(sum);
     	}
 	}
}


void convert_to_greyscale(int height, int width, uint8_t *img, uint8_t *grey_img)
{
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			auto b = img[i*width*3 + j*3 + 0];
			auto g = img[i*width*3 + j*3 + 1];
			auto r = img[i*width*3 + j*3 + 2];

			int average = (int)(0.3*r + 0.59*g + 0.11*b); // Adjust the weights for each channel

			grey_img[i*width + j] = average;
		}
	}
}

void compute_magnitude_and_gradient(int height, int width, uint8_t *Ix, uint8_t *Iy, uint8_t *mag, float *grad){
	
	for(int i = 1; i < height-1; i++)
 	{
     	for(int j = 1; j < width-1; j++)
	 	{
			float dx = Ix[i*width+j];
			float dy = Iy[i*width+j];
			mag[i*width+j] = round(sqrt(dx*dx+dy*dy)); // divde by sqrt(2) for scaling?
			float angle = atan2(dy, dx)*180/M_PI;
			grad[i*width+j] = angle < 0 ? angle+180 : angle;
     	}
 	}
}

void non_maximum_suppression(int height, int width, uint8_t *suppr_mag, uint8_t *mag, float* grad){

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			suppr_mag[i*width + j] = 0;
		}
	}


	for(int i = 1; i < height-1; i++)
	{
		for(int j = 1; j < width-1; j++)
		{
			int q = 255;
			int r = 255;

			//angle 0
			if (0 <= grad[i*width+j] < 22.5 || 157.5 <= grad[i*width+j] <= 180){
				q = mag[i*width + j+1];
				r = mag[i*width + j-1];
			}
			//angle 45
			else if (22.5 <= grad[i*width+j] < 67.5){
				q = mag[(i+1)*width + j-1];
				r = mag[(i-1)*width + j+1];
			}
			//angle 90
			else if (67.5 <= grad[i*width+j] < 112.5){
				q = mag[(i+1)*width + j];
				r = mag[(i-1)*width + j];
			}
			//angle 135
			else if (112.5 <= grad[i*width+j] < 157.5){
				q = mag[(i-1)*width + j-1];
				r = mag[(i+1)*width + j+1];
			}

			// Fine-tuning the threshold values
			// actually there shouldn't be a threshold here
			float threshold = NON_MAX_SUPPR_THRESHOLD; // Adjust this value to make the suppression less aggressive

			if (mag[i*width + j] >= q * threshold && mag[i*width + j] >= r * threshold){
				suppr_mag[i*width + j] = mag[i*width + j];
			} else {
				suppr_mag[i*width + j] = 0;
			}

		}
	}
}

void double_threshold(int height, int width, uint8_t *pixel_classification, uint8_t *suppr_mag) {
	float max_mag = 0.0;

	// Find the maximum and minimum magnitude values
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float mag = suppr_mag[i * width + j];
			max_mag = std::max(max_mag, mag);
			
		}
	}

	// Calculate the new threshold values
	float high_threshold = MAX_THRESHOLD_MULT * max_mag;
	float low_threshold = MIN_THRESHOLD_MULT * max_mag;

	printf("high_threshold: %f \n", high_threshold);
	printf("low_threshold: %f \n", low_threshold);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (suppr_mag[i * width + j] >= high_threshold) {
				// Strong pixels
				pixel_classification[i * width + j] = 255;
			} else if (suppr_mag[i * width + j] < low_threshold) {
				// Non-relevant pixels
				pixel_classification[i * width + j] = 0;
			} else {
				// Weak pixels
				pixel_classification[i * width + j] = 50;
			}
		}
	}
}

void hysteresis(int height, int width, uint8_t *pixel_classification){

	for(int i = 1; i < height-1; i++)
 	{
     	for(int j = 1; j < width-1; j++)
	 	{
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
}

void apply_dilation(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
        float max_val = std::numeric_limits<float>::min();

        for (int k = 0; k < kernel_size; k++) {
            for (int m = 0; m < kernel_size; m++) {
                float pixel_value = kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)];
                max_val = std::max(max_val, pixel_value);
            }
        }

        output[i * width + j] = max_val;
    }
}

}

void apply_erosion(int kernel_size, int height, int width, uint8_t *output, uint8_t *input, float *kernel)
{
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			float min_val = std::numeric_limits<float>::max();

			for (int k = 0; k < kernel_size; k++)
			{
				for (int m = 0; m < kernel_size; m++)
				{
					min_val = std::min(min_val, kernel[k * kernel_size + m] * input[(i + (k - 1)) * width + j + (m - 1)]);
				}
			}
			output[i * width + j] = (int)min_val;
		}
	}
}

void measure_time(bool start, FILE* file_times, std::string name){
	static std::chrono::system_clock::time_point start_time;
	static std::chrono::system_clock::time_point end_time;
	
	if(start){
		start_time = std::chrono::system_clock::now();
	} else {
		end_time = std::chrono::system_clock::now();
		std::chrono::duration<double> duration = end_time - start_time;
		fprintf(file_times, "%s: %f \n", name.c_str(), duration.count());
	}
}

float* get_gaussian_filter (int kernel_size, float sigma){
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
    int width, height, bpp;

	// does this datatype lead to bank conflicts? yes but we can't do anything about it
    
	auto img_fname = argc>=2 ? argv[1] : "image.png";

	system("mkdir -p output");
	auto file_times = fopen("./output/times.txt", "a");
	
	uint8_t* rgb_image = stbi_load(img_fname, &width, &height, &bpp, 3);

    std::cout<<"image: "<<img_fname<<std::endl;
	std::cout<<width<<" "<<height<<std::endl;

	// Convert to greyscale
    uint8_t* grey_image;
    grey_image = (uint8_t*)malloc(width*height);

	measure_time(true, file_times, "convert_to_greyscale");
	convert_to_greyscale(height, width, rgb_image, grey_image);
	measure_time(false, file_times, "convert_to_greyscale");
	stbi_image_free(rgb_image);

// 	Apply Gaussian filtering
// Choose the kernel size you want
	
	auto kernel_size = GAUSSIAN_KERNEL_SIZE;
	float sigma = GAUSSIAN_SIGMA;
	#if LAPLACIAN_GAUSSIAN
		float* gaussian_filter = get_gaussian_laplacian_filter(kernel_size, sigma);
	#else
		float* gaussian_filter = get_gaussian_filter(kernel_size, sigma);
	#endif

	for (int i = 0; i < kernel_size; i++){
		for (int j = 0; j < kernel_size; j++){
			printf("%f ", gaussian_filter[i*kernel_size + j]);
		}
		printf("\n");
	}

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

	uint8_t* dilation;
	dilation = (uint8_t*)malloc(width*height);

	//dilation kernel
	auto dilation_kernel_size = 3;
	float dilation_kernel[kernel_size*kernel_size] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

	measure_time(true, file_times, "dilation");
	apply_dilation(dilation_kernel_size, height, width, dilation, pixel_classification, dilation_kernel);
	measure_time(false, file_times, "dilation");

	stbi_write_png("./output/6_dilation.png", width, height, 1, dilation, width);

	uint8_t* erosion;
	erosion = (uint8_t*)malloc(width*height);

	//erosion kernel
	auto erosion_kernel_size = 4;
	float erosion_kernel[erosion_kernel_size*erosion_kernel_size] = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1
	};

	// auto erosion_kernel_size = 5;
	// float erosion_kernel[erosion_kernel_size*erosion_kernel_size] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	measure_time(true, file_times, "erosion");
	apply_erosion(erosion_kernel_size, height, width, erosion, dilation, erosion_kernel);
	measure_time(false, file_times, "erosion");

	

	stbi_write_png("./output/7_erosion.png", width, height, 1, erosion, width);



    return 0;
}