#ifndef UTILS_HPP
#define UTILS_HPP
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/imgcodecs.hpp"


cv::Mat TensorToMat (torch::Tensor tensor);

#endif