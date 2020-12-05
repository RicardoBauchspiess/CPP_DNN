#ifndef DATASET_HPP
#define DATASET_HPP

#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include <iostream>


class DataSet {
public:
	torch::Tensor getImage(int);
	torch::Tensor getLabel(int);
	int numSamples() {
		return labels.size();
	}
protected:
	// Dataset cannot be created on its own,
	// Inherited classes will create it
	DataSet(){};
	std::vector<torch::Tensor> labels;
	std::vector<torch::Tensor> images; 
private:
	
	
	

};
#endif