#include <iostream>
#include "utils.hpp"
#include "cifar10.h"
#include "SubsetRandomSampler.hpp"
#include "dataaugmentation.hpp"

using namespace torch;
using namespace cv;
using namespace std;


int main(int argc, char **argv) {

	if (argc < 2) {
		cout << "Pass data folder as second argument" << endl;
		return 0;
	}

	// Load CIFAR10 data
	DataSet* CIFAR10Data = CIFAR10::trainData(argv[1]);
	
	// Image transforms: random cropping and random horizontal flip
	auto transforms = Compose({pad({4,4,4,4},0), randomCrop(32,32), randomHorizontalFlip(0.5)});
	
	// create subset random sampler
	vector<int> idx(CIFAR10Data->numSamples());
	std::iota(begin(idx),end(idx),0);
	SubsetRandomSampler sampler(idx, CIFAR10Data, 10, transforms);
	
	// Show sampled transformed images
	int key = 0;
	while (key != 27) {
		
		int i = 0;

		auto batch = sampler.getBatch();

		auto samples = batch.first.split(1);
		while (key != 27 && i < samples.size() && key != 'n') {
			auto img = TensorToMat(samples[i]);
			resize(img,img,Size(400,400));
			imshow("img",img);
			key = waitKey(0);
			i++;
		}
		if (key == 27 ) {
			break;
		} else {
			key = 0;
		}
	}

	return 0;
}