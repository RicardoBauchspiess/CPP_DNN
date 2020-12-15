#include <iostream>
#include "utils.hpp"
#include "cifar10.h"
#include "SubsetRandomSampler.hpp"
#include "dataaugmentation.hpp"

using namespace torch;
using namespace cv;
using namespace std;


// Place holder net to develop code
struct Net : nn::Module {
	Net () {
		fc = register_module("fc1",nn::Linear(32*32*3,10));
	}

	Tensor forward(Tensor x) {
		return fc->forward(x.reshape({x.size(0), 32*32*3}));
	}

	nn::Linear fc{nullptr};
};


int main(int argc, char **argv) { 

	// training hyperparameters
	int epochs = 300;
	int batch_size = 128;
	int epoch_iter = 352;
	int iterations = epoch_iter*epochs;

	// training data
	DataSet* CIFAR10Data = CIFAR10::trainData(argv[1]);
	vector<int> idx(CIFAR10Data->numSamples());
	std::iota(begin(idx),end(idx),0);
	auto transforms = Compose({pad({4,4,4,4},0), randomCrop(32,32), randomHorizontalFlip(0.5)});
	SubsetRandomSampler sampler(idx, CIFAR10Data, batch_size, transforms);

	// model	
	auto net = std::make_shared<Net>();
	net->to(torch::kCUDA);
	
	// optimizer
	optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

	for (int i = 0; i < iterations; i++) {
		optimizer.zero_grad();
		auto batch = sampler.getBatch();
		auto images = batch.first.to(at::kCUDA);
		auto target = batch.second.to(at::kCUDA);
		
		auto prediction = net->forward(images);
		
		// Cross entropy loss
		auto loss = nll_loss(log_softmax(prediction,1), target);
		loss.backward();
		optimizer.step();
		torch::save(net,"net.pt");
		cout << "Loss: " << loss.item<float>() << endl;
		
	}


	return 0;
}