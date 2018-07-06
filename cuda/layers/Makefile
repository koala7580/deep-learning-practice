CXX := nvcc
TARGET := conv2d
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn
OPENCV_LIBS := -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

all: ${TARGET}

conv2d: conv2d.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) ${OPENCV_LIBS} $< -o ${TARGET}.out

run: ${TARGET}
	./${TARGET}.out

.phony: clean

clean:
	rm -f ${TARGET}.out || echo -n ""
