CC=g++
CUDAINCLUDE=/lihongliang/fangzl/cuda-12.4.1/include/
CUDALIB=/lihongliang/fangzl/cuda-12.4.1/lib64
CUDNNLIB=/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/venv/lib/python3.8/site-packages/nvidia/cudnn/lib

libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) -O3 -fPIC -shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -lcudart -lcublas -lcudnn -lpthread -o libinttemp.so

all:
	make libinttemp.so

clean:
	rm -rf *.o libinttemp.so
