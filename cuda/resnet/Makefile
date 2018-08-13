CXX := nvcc
TARGET := resnet
CUDNN_PATH := /usr/local/cuda
INCLUDE_DIRS := -I$(CUDNN_PATH)/include -I/usr/include/hdf5/serial -Isrc
LIBS := -L$(CUDNN_PATH)/lib64 -L/usr/local/lib -lcudnn -lcublas -lhdf5_cpp -lhdf5_serial -lm
CXXFLAGS := -std=c++14 -O2 -ccbin clang-3.9 -lstdc++

exception_test: $(patsubst %,build/%.o,exception catch exception_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cudnn_context_test: $(patsubst %,build/%.o,cudnn_context exception data_type catch cudnn_context_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

tensor_test: $(patsubst %,build/%.o,tensor array4d data_type exception catch tensor_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

kernel_test: $(patsubst %,build/%.o,kernel array4d data_type exception catch kernel_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

conv2d_test: $(patsubst %,build/%.o,cudnn_context tensor kernel array4d data_type exception conv2d catch conv1d_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

linear_test: $(patsubst %,build/%.o,cudnn_context tensor array4d data_type exception linear catch linear_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cuda_test: $(patsubst %,build/%.o,catch cuda_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cuda_test: $(patsubst %,build/%.o,catch cuda_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

cublas_test: $(patsubst %,build/%.o,catch cublas_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@

hdf5_test: $(patsubst %,build/%.o,cudnn_context data_type array4d exception tensor linear kernel conv2d catch hdf5_test)
	$(CXX) $(CXXFLAGS) $(LIBS) -o build/$@ $^
	./build/$@


build/%.o: tests/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

build/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c -o build/${@F} $<

.phony: clean

clean:
	rm -f build/*

