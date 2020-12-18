#include "models.hpp"

using namespace torch;
using namespace std;


// Standard Residual: 3x3-3x3 convolutions
Residual::Residual (int width) {
	
	residual->push_back(nn::Conv2d(nn::Conv2dOptions(width, width,3).stride(1).padding(1).bias(false) ));
	residual->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(width)));
	residual->push_back(nn::Functional(relu));
	residual->push_back(nn::Conv2d(nn::Conv2dOptions(width, width, 3).stride(1).padding(1).bias(false) ));
	residual->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(width)));

	for (int k = 0 ; k < residual->size(); k++) {
		residual[k]->to(torch::kCUDA);
	}
}

Tensor Residual::forward(Tensor x) {
	return torch::relu(x + residual->forward(x));
}

void Residual::sendToCUDA() {
	for (int k = 0 ; k < residual->size(); k++) {
		residual[k]->to(torch::kCUDA);
	}
}


// Residual residual with projection, stride=2
ResidualP::ResidualP (int input_width, int output_width) {
	
	residual->push_back(nn::Conv2d(nn::Conv2dOptions(input_width, output_width,3).stride(2).padding(1).bias(false) ));
	residual->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(output_width)));
	residual->push_back(nn::Functional(relu));
	residual->push_back(nn::Conv2d(nn::Conv2dOptions(output_width, output_width, 3).stride(1).padding(1).bias(false) ));
	residual->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(output_width)));

	projection->push_back(nn::AvgPool2d(nn::AvgPool2dOptions({2, 2}).stride({2, 2})) );
	projection->push_back(nn::Conv2d(nn::Conv2dOptions(input_width, output_width,1).stride(1).padding(0).bias(false) ));
	projection->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(output_width)));

	for (int k = 0 ; k < residual->size(); k++) {
		residual[k]->to(torch::kCUDA);
	}
	for (int k = 0 ; k < projection->size(); k++) {
		projection[k]->to(torch::kCUDA);
	}

}

Tensor ResidualP::forward(Tensor x) {
	return torch::relu(projection->forward(x) + residual->forward(x));
}

void ResidualP::sendToCUDA() {
	for (int k = 0 ; k < residual->size(); k++) {
		residual[k]->to(torch::kCUDA);
	}
	for (int k = 0 ; k < projection->size(); k++) {
		projection[k]->to(torch::kCUDA);
	}
}

// Classifier layers
Classifier::Classifier (int width, int classes) : avgpool(nn::AvgPool2dOptions(8).stride(8)), linear(nn::Linear(width,classes)) {
	this->width = width;

	avgpool->to(torch::kCUDA);
	linear->to(torch::kCUDA);
}

Tensor Classifier::forward(Tensor x) {
	return linear->forward((avgpool(x)).reshape({-1, width}));
}

void Classifier::sendToCUDA() {
	avgpool->to(torch::kCUDA);
	linear->to(torch::kCUDA);
}

// ResNet with original blocks + ResNet-D projection
ResNet::ResNet (int depth, int classes) {

	int stage_residuals = (depth-2)/6;

	vector<int> widths = {16, 32, 64};

	layers->push_back(nn::Conv2d(nn::Conv2dOptions(3, widths[0],3).stride(1).padding(1).bias(false) ));
	
	layers->push_back(nn::BatchNorm2d(nn::BatchNorm2dOptions(widths[0])));
	layers->push_back(nn::Functional(relu));

	for (int i = 0; i < widths.size(); i++) {
		int repeats;
		if (i == 0) {
			repeats = stage_residuals;
		} else {
			repeats = stage_residuals-1;
			layers->push_back(ResidualP(widths[i-1], widths[i]));
		}
		for (int j = 0; j < repeats; j++) {
			layers->push_back(Residual(widths[i]));
		}
	}

	layers->push_back(Classifier(widths[widths.size()-1], classes));


	for (int k = 0; k < layers->size(); k++) {
		if (k < 3) {
			layers[k]->to(torch::kCUDA);
		} 
		/*
		else {
			layers[k]->sendToCUDA();
		}
		*/
	}

}

Tensor ResNet::forward(Tensor x) {
	return layers->forward(x);
}

void ResNet::sendToCUDA() {
	int k = 0;

	for (int k = 0; k < layers->size(); k++) {
		if (k < 3) {
			layers[k]->to(torch::kCUDA);
		} 
		/*
		else {
			layers[k]->sendToCUDA();
		}
		*/
	}
}