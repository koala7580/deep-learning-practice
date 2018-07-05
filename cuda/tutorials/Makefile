CXX := nvcc
NVCC_OPTS := --std c++14 -ccbin clang-3.9 -lstdc++

all: cuda_add

cuda_add:
	${CXX} ${NVCC_OPTS} -o cuda_demo cuda.cu

.PHONY: clean
clean:
	rm -f cuda_demo