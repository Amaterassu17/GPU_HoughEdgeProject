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
    

}