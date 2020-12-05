#include "DataSet.hpp"

using namespace torch;


Tensor DataSet::getImage(int idx) {
	return images[idx];
}

Tensor DataSet::getLabel(int idx) {
	return labels[idx];
}