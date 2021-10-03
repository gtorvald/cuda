#include <iostream>
#include <cmath>
#include "lib/lodepng.h"
#include <vector>
#include <exception>
#include <stdexcept>

#define GRID_SIZE 5

__global__ void  doFilter (
		unsigned char *img,
		unsigned char *new_img,
		unsigned int width,
		unsigned int height,
		double *filter_matrix,
		unsigned char filter_dim
	) {
	// инициализация индексов
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	// инициализация вспомогательных переменных для обработки краевых значений
	int begin = filter_dim / 2;
	int end_offset = filter_dim / 2;
	
	// индексы за пределами массива изображения
	if (x >= width || y >= height) {
		return;
	// фильтрация изображения
	} else if ((x >= begin && x < width - end_offset) &&
			(y >= begin && y < height - end_offset)) {
			double new_pxl = 0;
			for (int i = 0; i < filter_dim; ++i) {
				for (int j = 0; j < filter_dim; ++j) {
					new_pxl += img[3 * ((x - 1 + i) * width + (y - 1 + j)) + z] * 
												filter_matrix[i * filter_dim + j];
				}
			}
			new_img[3 * (x * width + y) + z] = min(255., max(0., new_pxl));
	// края изображения окрашиваются в черный
	} else if (x < begin || x >= width - end_offset ||
			y < begin || y >= height - end_offset) {
		new_img[3 * (x * width + y) + z] = 0;
	}
}

// структура фильтра
struct Filter {
	double *matrix;
	unsigned char dim;
};

// определение фильтра Gaussian blur
Filter getGaussianBlur() {
	Filter filter;
	filter.dim = 5;

	double host_filter[25] = {1, 4, 6, 4, 1,
														4, 16, 24, 16, 4,
														6, 24, 36, 24, 6,
														4, 16, 24, 16, 4,
														1, 4, 6, 4, 1};
	unsigned char filter_len = 25;
	for (int i = 0; i < filter_len; ++i) {
		host_filter[i] /= 256.;
	}
	size_t bytes = 25 * sizeof(double);
	cudaMalloc(&filter.matrix, bytes);
	cudaMemcpy(filter.matrix, host_filter, bytes, cudaMemcpyHostToDevice);

	return filter;
}

// определение фильтра Edge detection
Filter getEdgeDetection() {
	Filter filter;
	filter.dim = 3;

	double host_filter[9] = {-1, -1, -1,
													-1, 8, -1,
													-1, -1, -1};
	size_t bytes = 9 * sizeof(double);
	cudaMalloc(&filter.matrix, bytes);
	cudaMemcpy(filter.matrix, host_filter, bytes, cudaMemcpyHostToDevice);

	return filter;
}

// определение фильтра Sharpen
Filter getSharpen() {
	Filter filter;
	filter.dim = 3;

	double host_filter[9] = {0, -1, 0,
													-1, 5, -1,
													0, -1, 0};
	size_t bytes = 9 * sizeof(double);
	cudaMalloc(&filter.matrix, bytes);
	cudaMemcpy(filter.matrix, host_filter, bytes, cudaMemcpyHostToDevice);

	return filter;
}

// валидация аргументов и определение фильтра
Filter getFilter(int argc, char** argv) {
	if (argc < 2) {
		throw std::invalid_argument("error: not filter detected");
	}

	std::string filter(argv[1]);
	if (filter == "blur") {
		return getGaussianBlur();
	} else if (filter == "edge") {
		return getEdgeDetection();
	} else if (filter == "sharpen") {
		return getSharpen();
	}
	throw std::invalid_argument("error: invalid filter");
}

// ./main <filter> <image> [<image>]
int main(int argc, char **argv) {
	// валидация аргументов и определение фильтра
	Filter filter;
	try {
		filter = getFilter(argc, argv);
		if (argc < 3) {
			throw std::invalid_argument("error: not files detected");
		}
	} catch(std::invalid_argument ex) {
		std::cout << ex.what() << std::endl;
		return 0;
	}

	// для среднего времени выполнения
	int count = 0;
	float ms_all = 0, ms_kernel_all = 0;

	// цикл обработки изображений из параметров
	for (int i = 2; i < argc; ++i) {
		// инициализация переменных для подсчета времени
		cudaEvent_t start, stop, start_kernel, stop_kernel;
		cudaEventCreate(&start_kernel);
		cudaEventCreate(&start);
		cudaEventCreate(&stop_kernel);
		cudaEventCreate(&stop);

		// считывание картинки из файла на хост
		std::string file(argv[i]);
		unsigned char *host_img;
		unsigned int width, height;
		if (lodepng_decode24_file(&host_img, &width, &height, file.data())) {
				std::cout << "error: image " << file << " wasn't decoded correctly" << std::endl;
				continue;
		}
		size_t img_size = width * height * 3;

		// копирование картинки с хоста на девайс
		cudaEventRecord(start);
		size_t bytes = img_size * sizeof(unsigned char);    
		unsigned char *device_img;
		unsigned char *device_new_img;
		cudaMalloc(&device_img, bytes);
		cudaMalloc(&device_new_img, bytes);
		cudaMemcpy(device_img, host_img, bytes, cudaMemcpyHostToDevice);

		// расчет размера блока
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, 0);
		int size = floor(sqrt(device_prop.maxThreadsPerBlock) / 3);

		// запуск вычислений на GPU
		dim3 blockDims(size, size, 3);
		dim3 gridDims(ceil(width / (double) size), ceil(height / (double) size), 1);
		cudaEventRecord(start_kernel);
		doFilter<<<gridDims, blockDims>>>(device_img, device_new_img, width, height,
																			filter.matrix, filter.dim);
		cudaDeviceSynchronize();
		cudaEventRecord(stop_kernel);
		
		// копирование отфильтрованной картинки с девайса на хост
		unsigned char *host_new_img = new unsigned char[img_size];
		cudaMemcpy(host_new_img, device_new_img, bytes, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaEventRecord(stop);

		// запись отфильтрованной картинки в файл
		std::string new_file("new_" + file);
		if (lodepng_encode24_file(new_file.data(), host_new_img, width, height)) {
			std::cout << "error: image " << file << " wasn't encoded correctly" << std::endl;
		} else {
			std::cout << "success: filtered image " << file << " now in " << new_file << std::endl;
		}
		
		cudaFree(device_img);
		cudaFree(device_new_img);
		free(host_img);
		delete [] host_new_img;

		// вывод времени
		cudaEventSynchronize(stop);
		cudaEventSynchronize(stop_kernel);
		float ms, ms_kernel;
		cudaEventElapsedTime(&ms, start, stop);
		cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel);
		std::cout << "\tkernel time:\t" << ms_kernel << " ms" << std::endl;
		std::cout << "\tprog time:\t" << ms << " ms" << std::endl;
		ms_all += ms;
		ms_kernel_all += ms_kernel;
		++count;
	}

	// вывод среднего времени работы
	std::cout << "kernel time:\t" << ms_kernel_all / count << " ms" << std::endl;
	std::cout << "prog time:\t" << ms_all / count << " ms" << std::endl;
	
	cudaFree(filter.matrix);
  return 0;
}
