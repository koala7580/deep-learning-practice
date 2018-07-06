#include <iostream>
#include <string>
#include <stdexcept>
#include <cudnn.h>
#include <opencv2/opencv.hpp>

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

void save_image(const char* output_filename,
    float* buffer,
    int height,
    int width) {
    cv::Mat output_image(height, width, CV_32FC3, buffer);

    // Make negative values zero.
    cv::threshold(output_image,
        output_image,
        /*threshold=*/0,
        /*maxval=*/0,
        cv::THRESH_TOZERO);

    cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8UC3);
    cv::imwrite(output_filename, output_image);
}

namespace cudnn {
    void _checkStatus(cudnnStatus_t status, const std::string &fnName) {
        if (status != CUDNN_STATUS_SUCCESS) {
            std::stringstream ss;
            ss << "Error when calling " << fnName << ": "
                << cudnnGetErrorString(status) << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    class CuDNNContext {
        cudnnHandle_t _context;
    public:
        CuDNNContext() {
            _checkStatus(cudnnCreate(&_context), "cudnnCreate");
        }
        ~CuDNNContext() {
            cudnnDestroy(_context);
        }
        explicit operator cudnnHandle_t() const noexcept { return _context; }

        CuDNNContext(CuDNNContext const &) = delete;
        CuDNNContext& operator=(CuDNNContext const &) = delete;
        CuDNNContext(CuDNNContext &&) = delete;
    };

    class Tensor {
        cudnnTensorDescriptor_t _descriptor;
    public:
        Tensor() {

        }
        ~Tensor() {
            cudnnDestroyTensorDescriptor(_descriptor);
        }
        explicit operator cudnnTensorDescriptor_t() const noexcept { return _descriptor; }

        Tensor(Tensor const &) = delete;
        Tensor& operator=(Tensor const &) = delete;
        Tensor(Tensor &&) = delete;
    };

    class Conv2D {
    public:
        Conv2D() {

        }
        ~Conv2D() {

        }

        Conv2D(Conv2D const &) = delete;
        Conv2D& operator=(Conv2D const &) = delete;
        Conv2D(Conv2D &&) = delete;
    };

    class DeviceMemoryManager {
    public:
        DeviceMemoryManager() {

        }
        ~DeviceMemoryManager() {

        }

        DeviceMemoryManager(DeviceMemoryManager const &) = delete;
        DeviceMemoryManager& operator=(DeviceMemoryManager const &) = delete;
        DeviceMemoryManager(DeviceMemoryManager &&) = delete;
    };
}

int main(int argc, char** argv) {
    return 0;
}