#include "SubsetRandomSampler.hpp"

using namespace std;


SubsetRandomSampler::SubsetRandomSampler(vector<int> indexes, DataSet* dataset, int batchsize) 
    : mMersenneEngine((std::random_device())() ), mDist (0,(int)indexes.size()-1)
{

	mIndexes = indexes;
	mDataSet = dataset;
	mBatchSize = batchsize;

    mBatchIndexes = vector<int>(mBatchSize);
    mBatchImages = vector<torch::Tensor>(mBatchSize);
    mBatchLabels = vector<torch::Tensor>(mBatchSize);
    
    // Creates random number generators
    mRandomGen = [this](){
        return mIndexes[mDist(mMersenneEngine)];
        };

    mSampleIdx = -1;

    mSampleImage = [this](){
        mSampleIdx++;
        return mDataSet->getImage(mIndexes[mBatchIndexes[mSampleIdx]] );
        };

    mSampleLabel = [this](){
        mSampleIdx++;
        return mDataSet->getLabel(mIndexes[mBatchIndexes[mSampleIdx]] );
        };

}

std::pair<torch::Tensor, torch::Tensor> SubsetRandomSampler::getBatch() {

    
    generate(begin(mBatchIndexes), end(mBatchIndexes), mRandomGen);

    //mBatchImages;
    mSampleIdx = -1;
    generate(begin(mBatchImages), end(mBatchImages), mSampleImage);

    //mBatchLabels;
    mSampleIdx = -1;
    generate(begin(mBatchLabels), end(mBatchLabels), mSampleLabel);

    pair<torch::Tensor, torch::Tensor> batch(torch::stack(mBatchImages),torch::stack(mBatchLabels));



    return batch;
}