#ifndef SKEL_CUDA_HPP
#define SKEL_CUDA_HPP
#include "field.h"

FIELD<float>* computeSkeleton(int level, FIELD<float> *im);
int initialize_skeletonization(FIELD<float>* im);
void deallocateCudaMem();
short* get_current_skel_ft();
FIELD<float>* skelft_to_field();

#endif