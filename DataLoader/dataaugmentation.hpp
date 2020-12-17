#ifndef DATAAUGMENTATION_HPP
#define DATAAUGMENTATION_HPP
#include <torch/torch.h>

#include <random>



/** randomHorizontalFlip
 *	\brief create random horizontal flip function
 *
 */
std::function<torch::Tensor(torch::Tensor)> randomHorizontalFlip(float prob);

/** randomCrop
 *	\brief create random crop function
 *
 */
std::function<torch::Tensor(torch::Tensor)> randomCrop(int h, int w);


/** Compose
 *	\brief combine functions sequentially
 *
 */
std::function<torch::Tensor(torch::Tensor)> Compose(std::vector<std::function<torch::Tensor(torch::Tensor)> > functions);

/** Pad
 *	\brief pad
 *
 */
std::function<torch::Tensor(torch::Tensor)> pad(torch::IntArrayRef pad, torch::Scalar value );

/** Normalize
 *	\brief normalize tensor
 *
 */
std::function<torch::Tensor(torch::Tensor)> normalize(std::vector<double> mean, std::vector<double> std );

#endif