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


/** CombineFunctions
 *  \brief create a function as a sequence of two functions
 *
 */
std::function<torch::Tensor(torch::Tensor)> CombineFunctions(std::function<torch::Tensor(torch::Tensor)> f1, 
	std::function<torch::Tensor(torch::Tensor)> f2) {

	return [f1, f2](torch::Tensor input){
		return f2(f1(input));
	};
}

/** Compose
 *  \brief create sequention of transform functions
 *
 */
std::function<torch::Tensor(torch::Tensor)> Compose(std::vector<std::function<torch::Tensor(torch::Tensor)> > functions) {

	if (functions.size() > 1) {
		auto function = CombineFunctions(functions[0],functions[1]);
		for (int i = 2; i < functions.size(); i++ ) {
			function = CombineFunctions(function,functions[i]);
		}
		return function;
	} else if (functions.size() == 1) {
		return functions[0];
	}	else {
		// return identity function
		return [](torch::Tensor input){
			return input;
		}; 
	}


}

/** Pad
 *	\brief pad
 *
 */
std::function<torch::Tensor(torch::Tensor)> pad(torch::IntArrayRef pad, torch::Scalar value = 0) {
	return [pad, value](torch::Tensor input){
		return torch::constant_pad_nd(input, pad, value);
		//torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({1,2,3}).mode(pad_mode));
	};
}