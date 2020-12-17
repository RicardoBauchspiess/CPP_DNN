#ifndef CIFAR10_HPP
#define CIFAR10_HPP
#include "DataSet.hpp"


#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "dataaugmentation.hpp"

#include <iostream>

// Test set and Train set as singletons, so that the data won't be loaded multiple times
class CIFAR10 : public DataSet {
public:
	static CIFAR10* trainData(const std::string& root, std::function<torch::Tensor(torch::Tensor)> transforms = Compose({})) {
        static CIFAR10 *mTrainInstance = 0;
        if(mTrainInstance == 0)
        {
            mTrainInstance = new CIFAR10(root, true, transforms);
        }
        return mTrainInstance;
    }

    static CIFAR10* testData(const std::string& root, std::function<torch::Tensor(torch::Tensor)> transforms = Compose({})) {
        static CIFAR10 *mTestInstance = 0;
        if(mTestInstance == 0)
        {
            mTestInstance = new CIFAR10(root, false, transforms);
        }
        return mTestInstance;
    }

private:

	CIFAR10(const std::string& root, bool train, std::function<torch::Tensor(torch::Tensor)> transforms);
 


};
#endif