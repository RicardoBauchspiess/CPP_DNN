#ifndef DATAAUGMENTATION_HPP
#define DATAAUGMENTATION_HPP
#include <torch/torch.h>

#include <random>



/** \brief create random horizontal flip function
 *
 */
std::function<torch::Tensor(torch::Tensor)> randomHorizontalFlip(float prob);


std::function<torch::Tensor(torch::Tensor)> randomCrop(int h, int w);

#endif