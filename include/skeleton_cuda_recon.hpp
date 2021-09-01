#ifndef SKEL_CUDA_HPP
#define SKEL_CUDA_HPP
#include "field.h"

FIELD<float>* computeCUDADT(FIELD<float> *im);
int initialize_skeletonization_recon(int xM, int yM);
void deallocateCudaMem_recon();

#endif
