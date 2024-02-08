#include <stdint.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>


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


int  main(int argc, char *argv[]){

    int width, height, channels;
    auto img_fname= argc>=2 ? argv[1] : "images/lena.png";

    system("mkdir -p output_Hough");
    auto file_times = fopen("output_Hough/times.txt", "w");

    uint8_t *img = stbi_load("images/lena.png", &width, &height, &channels, 0);

    std::cout << "Image size: " << width << "x" << height << "x" << channels << std::endl;


	//HOUGH TRANSFORM
	measure_time(true, file_times, "Hough Transform");
	int max_rho = (int)std::sqrt(width*width + height*height);
	int max_theta = 180;
	int *hough_space = new int[max_rho*max_theta];
	std::fill(hough_space, hough_space + max_rho*max_theta, 0);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			if(img[(y*width + x)*channels] < 128){
				for(int theta=0; theta<max_theta; theta++){
					float rho = x*std::cos(theta*M_PI/180) + y*std::sin(theta*M_PI/180);
					int rho_idx = (int)rho + max_rho/2;
					hough_space[rho_idx*max_theta + theta]++;
				}
			}
		}
	}
	measure_time(false, file_times, "Hough Transform");

	//FIND MAX
	measure_time(true, file_times, "Find Max");
	int max_hough = *std::max_element(hough_space, hough_space + max_rho*max_theta);
	measure_time(false, file_times, "Find Max");

	//DRAW LINES
	measure_time(true, file_times, "Draw Lines");
	uint8_t *img_hough = new uint8_t[width*height*channels];
	std::copy(img, img + width*height*channels, img_hough);
	for(int rho_idx=0; rho_idx<max_rho; rho_idx++){
		for(int theta=0; theta<max_theta; theta++){
			if(hough_space[rho_idx*max_theta + theta] > max_hough*0.5){
				float rho = rho_idx - max_rho/2;
				for(int x=0; x<width; x++){
					int y = (rho - x*std::cos(theta*M_PI/180))/std::sin(theta*M_PI/180);
					if(y>=0 && y<height){
						img_hough[(y*width + x)*channels] = 255;
						img_hough[(y*width + x)*channels + 1] = 0;
						img_hough[(y*width + x)*channels + 2] = 0;
					}
				}
				for(int y=0; y<height; y++){
					int x = (rho - y*std::sin(theta*M_PI/180))/std::cos(theta*M_PI/180);
					if(x>=0 && x<width){
						img_hough[(y*width + x)*channels] = 255;
						img_hough[(y*width + x)*channels + 1] = 0;
						img_hough[(y*width + x)*channels + 2] = 0;
					}
				}
			}
		}
	}
	measure_time(false, file_times, "Draw Lines");

	//SAVE IMAGE

	stbi_write_png("output_Hough/lena_hough.png", width, height, channels, img_hough, width*channels);

	fclose(file_times);
	stbi_image_free(img);
	delete[] hough_space;
	delete[] img_hough;
	return 0;
	
    

}