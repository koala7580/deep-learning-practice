CXX := nvcc
TARGET_CUDA := cuda_demo
TARGET_CUDNN := cudnn_demo
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET_CUDA} ${TARGET_CUDNN}


cuda_demo: cuda.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $< -o ${TARGET_CUDA}.out

cudnn_demo: cudnn.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) ${OPENCV_LIBS} $< -o ${TARGET_CUDNN}.out 

.phony: clean

clean:
	rm -f ${TARGET_CUDA}.out ${TARGET_CUDNN}.out || echo -n ""
