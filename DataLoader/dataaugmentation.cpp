#include "dataaugmentation.hpp"

#include <iostream>
using namespace std;

/** randomHorizontalFlip
 *  \brief create random horizontal flip function
 *
 */
std::function<torch::Tensor(torch::Tensor)> randomHorizontalFlip(float prob) {
	return [prob](torch::Tensor input){
		static mt19937 rng((std::random_device())() );
    	static uniform_real_distribution<> u(0.0, 1.0);
		static float fp = prob;

		if (u(rng) < fp) {
			return input.flip(3);
		} else {
			return input;
		}
	};
}


/** randomCrop
 *  \brief create random crop function
 *
 */
std::function<torch::Tensor(torch::Tensor)> randomCrop(int h, int w) {
	return [h, w](torch::Tensor input){
		static mt19937 rng((std::random_device())() );

    	uniform_int_distribution<> uw(0.0, input.size(2)-h);
    	uniform_int_distribution<> uh(0.0, input.size(3)-w);

    	int x = uw(rng);
    	int y = uh(rng);
    	return input.slice(2,y,y+h,1).slice(3,x,x+w,1);

	};
}
