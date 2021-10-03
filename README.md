# Задание 1

Компиляция:

nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 main.cu lib/lodepng.cpp -o main
