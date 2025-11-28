CC=g++
NVCC=/lihongliang/fangzl/cuda-12.4.1/bin/nvcc
CFLAGS=-O3
CUDAINCLUDE=/lihongliang/fangzl/cuda-12.4.1/include/
CUDALIB=/lihongliang/fangzl/cuda-12.4.1/lib64
CUDNNLIB=/lihongliang/fangzl/orion-work/orion_gpt/gpt-example/venv/lib/python3.8/site-packages/nvidia/cudnn/lib

utils_sched.o: utils_sched.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC -x cu -shared -c utils_sched.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler.o: scheduler.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler_eval.o: scheduler_eval.cpp
	$(NVCC) $(CFLAGS) -Xcompiler  -fPIC  -x cu -shared -c scheduler_eval.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lpthread

scheduler_eval.so: scheduler_eval.o utils_sched.o
	 $(CC) $(CFLAGS) -fPIC -shared utils_sched.o scheduler_eval.o -I$(CUDAINCLUDE) -L$(CUDALIB) -L$(CUDNNLIB) -lcudart -lcudnn -lcublas -lpthread -o scheduler_eval.so

all:
	make scheduler_eval.so

clean:
	rm -rf *.o *.so
