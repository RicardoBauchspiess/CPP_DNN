#ifndef MODELS_HPP
#define MODELS_HPP
#include <torch/torch.h>

// Standard residual block: 3x3-3x3 convolutions
struct Residual : torch::nn::Module {
	Residual (int width);
	torch::Tensor forward(torch::Tensor x);
	void sendToCUDA();

private:
	torch::nn::Sequential residual;
};


// Standard residual block with projection, stride = 2
struct ResidualP : torch::nn::Module {
	ResidualP (int input_width, int output_width);
	torch::Tensor forward(torch::Tensor x);
	void sendToCUDA();

private:
	torch::nn::Sequential residual;
	torch::nn::Sequential projection;
};

// Classifier Layer
struct Classifier : torch::nn::Module {
	Classifier (int width, int classes);
	torch::Tensor forward(torch::Tensor x);
	void sendToCUDA();

private:
	torch::nn::AvgPool2d avgpool;
	torch::nn::Linear linear;
	int width;
};


// ResNet with original blocks + ResNet-D projection
struct ResNet : torch::nn::Module {
	ResNet (int depth, int classes);
	torch::Tensor forward(torch::Tensor x);
	void sendToCUDA();

private:
	torch::nn::Sequential layers;
};


#endif