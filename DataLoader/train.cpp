#include <iostream>
#include "utils.hpp"
#include "cifar10.h"
#include "SubsetRandomSampler.hpp"
#include "dataaugmentation.hpp"

#include "models.hpp"

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
	auto data_transforms = Compose({normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}), pad({4,4,4,4},0)});
        DataSet* CIFAR10Data = CIFAR10::trainData(argv[1],data_transforms);
	vector<int> idx(CIFAR10Data->numSamples());
	std::iota(begin(idx),end(idx),0);
	auto sampler_transforms = Compose({randomCrop(32,32), randomHorizontalFlip(0.5)});
	SubsetRandomSampler sampler(idx, CIFAR10Data, batch_size, sampler_transforms);

	// model	
	//auto net = std::make_shared<Net>();
	//net->to(torch::kCUDA);
	
	// net->to(torch::CUDA) didn't work, sending to CUDA during initialization instead

	auto net = std::make_shared<ResNet>(20,10);

	
	//net->sendToCUDA();
	//net->to(torch::kCUDA);

	// optimizer
	optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.2).momentum(0.9).weight_decay(0.0001).nesterov(true) );

	for (int i = 0; i < epochs; i++) {
		float num_samples = 0;
		float correct = 0;
		for (int j = 0;  j < epoch_iter; j++) {
			cout << j << endl;
			optimizer.zero_grad();
			auto batch = sampler.getBatch();
			auto images = batch.first.to(torch::kCUDA);
			auto target = batch.second.to(torch::kCUDA);	
			auto prediction = net->forward(images);
		
			//cout << "prediction made" << endl;
			// Cross entropy loss
			auto loss = nll_loss(log_softmax(prediction,1), target);
			loss.backward();
			optimizer.step();
			
			

			// Calculate accuracy
			auto vals = torch::argmax(prediction, 1);
			int k = 0;
			for (k = 0; k < vals.size(0); k++) {
				//cout << k << endl;
				if (vals[k].item<long>() == (int)target[k].item<long>()) {
					correct += 1;
				}
			}
			num_samples += (float)k;
			

		}
		cout << "Epoch " << i << ":  " << 100.0*correct/(num_samples+0.000001) << " percent correct" << endl;
		
		torch::save(net,"net.pt");
		torch::save(optimizer, "optimizer.pt");
		//cout << "Loss: " << loss.item<float>() << endl;
		
	}


	return 0;
}
