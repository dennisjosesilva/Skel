#ifndef SKEL_CUDA_HPP
#define SKEL_CUDA_HPP
#include "field.h"

FIELD<float>* computeCUDADT(FIELD<float> *im);
FIELD<float>* CUDA_interp(FIELD<float> *curr_l, FIELD<float> *prev_l, FIELD<float> *prev_d, FIELD<float> *curr_d, int curr_bound_value, int prev_bound_value, bool firstL, int lastL);

int initialize_skeletonization_recon(int xM, int yM);
void deallocateCudaMem_recon();

#endif
