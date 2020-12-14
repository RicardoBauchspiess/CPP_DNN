#ifndef SUBSETRANDOMSAMPLER_HPP
#define SUBSETRANDOMSAMPLER_HPP

#include <torch/torch.h>
#include "DataSet.hpp"


class SubsetRandomSampler {

public:
	
	SubsetRandomSampler(std::vector<int> indexes, DataSet* dataset, int batchsize,
		std::function<torch::Tensor(torch::Tensor)> transforms);

	std::pair<torch::Tensor, torch::Tensor> getBatch();
private:
	// Data
	int mBatchSize;
	DataSet* mDataSet;
	std::vector<int> mIndexes;
	std::vector<int> mBatchIndexes;
	std::vector<torch::Tensor> mBatchImages;
	std::vector<torch::Tensor> mBatchLabels;

	// Transforms
	std::function<torch::Tensor(torch::Tensor)> mTransforms;

	// Random number generator
	std::mt19937 mMersenneEngine;
	std::uniform_int_distribution<int> mDist;
	std::function<int()> mRandomGen;
	std::function<torch::Tensor()> mSampleImage;
	std::function<torch::Tensor()> mSampleLabel;

	int mSampleIdx;

};

#endif