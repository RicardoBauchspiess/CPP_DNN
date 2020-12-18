#include "utils.hpp"

using namespace std;
using namespace cv;
using namespace torch;

Mat TensorToMat (Tensor tensor) {

	vector<Mat> channels;
	int batch = tensor.size(0);
	int num_channels = tensor.size(1);
	int height = tensor.size(2);
	int width = tensor.size(3);


	auto out_tensor = tensor.view({num_channels,height,width}).permute({1,2,0});
	out_tensor = out_tensor.mul(255).clamp(0,255).to(torch::kU8);
	
	// OpenCV expects alternating B,G,R values
	// Tensor has all R values, followed by all G values and B values
	// Load each channels into a Mat, then merge those channels into a 3 channes image
	cv::Mat resultImg1(width, height,CV_8UC1, out_tensor.data_ptr());
	if (num_channels > 1) {
		int channels_size = height*width;
		cv::Mat resultImg2(width, height, CV_8UC1, out_tensor.data_ptr()+channels_size);
		cv::Mat resultImg3(width, height, CV_8UC1, out_tensor.data_ptr()+channels_size);
		Mat resultImg;
		channels = {resultImg3,resultImg2,resultImg1};
		merge(channels, resultImg);

		return resultImg;
	} else {
		return resultImg1;
	}

}