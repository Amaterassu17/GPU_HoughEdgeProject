// Edge detection

#include <stdint.h>
#include <iostream>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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
					sum += kernel[k*kernel_size + m]*input[(i+(k-1))*width + j + (m-1)];
				}
			}

			output[i*width + j] = sum;
     	}
 	}
}

int main()
{
    int width, height, bpp;

	// does this datatype lead to bank conflicts?
    uint8_t* rgb_image = stbi_load("esa.png", &width, &height, &bpp, 3);

    std::cout<<width<<" "<<height;

	// Convert to greyscale
    uint8_t* grey_image;
    grey_image = (uint8_t*)malloc(width*height);

	for(int i = 0; i < height; i++)
 	{
     	for(int j = 0; j < width; j++)
	 	{
			auto b = rgb_image[i*width*3 + j*3 + 0];
			auto g = rgb_image[i*width*3 + j*3 + 1];
			auto r = rgb_image[i*width*3 + j*3 + 2];

			int average = (int)(0.2126*r+0.7152*g+0.0722*b);

			grey_image[i*width + j] = average;
     	}
 	}

	 stbi_image_free(rgb_image);

	// Apply Gaussian filtering
	int kernel_size = 3;
	float gaussian_filter[kernel_size*kernel_size] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
	uint8_t* gaussian_image;
    gaussian_image = (uint8_t*)malloc(width*height);

	apply_filter(kernel_size, height, width, gaussian_image, grey_image, gaussian_filter);

	stbi_image_free(grey_image);
	stbi_write_png("image_gaussian.png", width, height, 1, gaussian_image, width);
	
	// Apply Sobel filtering
	float sobel_h[kernel_size*kernel_size] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float sobel_v[kernel_size*kernel_size] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	uint8_t* sobel_image_h;
    sobel_image_h = (uint8_t*)malloc(width*height);
	uint8_t* sobel_image_v;
    sobel_image_v = (uint8_t*)malloc(width*height);

	apply_filter(kernel_size, height, width, sobel_image_h, gaussian_image, sobel_h);
	apply_filter(kernel_size, height, width, sobel_image_v, gaussian_image, sobel_v);

	std::cout<<"end"<<" ";
   
    stbi_image_free(gaussian_image);
	stbi_write_png("image_sobel_h.png", width, height, 1, sobel_image_h, width);
	stbi_write_png("image_sobel_v.png", width, height, 1, sobel_image_v, width);

	// Calculate magnitude and gradient direction
    float* gradient_direction;
    gradient_direction = (float*)malloc(width*height*sizeof(float));
	float* magnitude;
    magnitude = (float*)malloc(width*height*sizeof(float));

	for(int i = 1; i < height-1; i++)
 	{
     	for(int j = 1; j < width-1; j++)
	 	{
			float dx = sobel_image_h[i*width+j];
			float dy = sobel_image_v[i*width+j];
			magnitude[i*width+j] = (float)sqrt(dx*dx+dy*dy);
			std::cout<<sqrt(dx*dx+dy*dy)<<" ";
			if(dy!=0 && dx != 0){
				gradient_direction[i*width+j] = atan(std::max(dy/dx, (float)10.0));
			} else {
				gradient_direction[i*width+j] = (float)0.0;
			}
     	}
 	}

	stbi_write_png("gradient_direction.png", width, height, 1, gradient_direction, width);
	stbi_write_png("magnitude.png", width, height, 1, magnitude, width);

    return 0;
}
