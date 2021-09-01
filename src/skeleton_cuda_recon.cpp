/* skeleton.cpp */

#include <math.h>
#include "fileio.hpp"
#include <cuda_runtime_api.h>
#include "genrl.h"
#include "skelft.h"
#include "skeleton_cuda_recon.hpp"

#define INDEX(i,j) (i)+fboSize_*(j)
float* siteParam_recon;

short* outputFT_recon;
//bool* foreground_mask;
int xm_ = 0, ym_ = 0, xM_, yM_, fboSize_;

void allocateCudaMem_recon(int size) {
    skelft2DInitialization(size);
    cudaMallocHost((void**)&outputFT_recon, size * size * 2 * sizeof(short));
    //cudaMallocHost((void**)&foreground_mask, size * size * sizeof(bool));
    cudaMallocHost((void**)&siteParam_recon, size * size * sizeof(float));
}

void deallocateCudaMem_recon() {
    skelft2DDeinitialization();
    cudaFreeHost(outputFT_recon);
    //cudaFreeHost(foreground_mask);
    cudaFreeHost(siteParam_recon);
}

int initialize_skeletonization_recon(int xM, int yM) {
    fboSize_ = skelft2DSize(xM, yM); //   Get size of the image that CUDA will actually use to process our nx x ny image
    allocateCudaMem_recon(fboSize_);
    return fboSize_;
}

// TODO(maarten): dit moet beter kunnen, i.e. in een keer de DT uit cuda halen
void dt_field(FIELD<float>* f) {
    for (int i = 0; i < xM_; ++i) {
        for (int j = 0; j < yM_; ++j) {
            int id = INDEX(i, j);
            int ox = outputFT_recon[2 * id];
            int oy = outputFT_recon[2 * id + 1];
            float val = sqrt((i - ox) * (i - ox) + (j - oy) * (j - oy));
            //if (foreground_mask[INDEX(i, j)]) f->set(i, j, val);
            //else f->set(i, j, 0);
            f->set(i, j, val);
        }
    }
}


FIELD<float>* computeCUDADT(FIELD<float> *input) {
    
    memset(siteParam_recon, 0, fboSize_ * fboSize_ * sizeof(float));
    //memset(foreground_mask, false, fboSize_ * fboSize_ * sizeof(bool));
    
    int nx = input->dimX();
    int ny = input->dimY();
    xm_ = ym_ = nx; xM_ = yM_ = 0;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (!(*input)(i, j)) {
                //sign = 1;// if the first layer is all-1, then it never go into here, so xm will larger than xM, that's where we have a problem in skelft2DMakeBoundary() fucntion. So I added a sign here.
                //foreground_mask[INDEX(i, j)] = true;
                siteParam_recon[INDEX(i, j)] = 1;
                xm_ = min(xm_, i); ym_ = min(ym_, j);
                //xM_ = max(xM_, i); yM = max(yM, j);
            }
        }
    }
    
    xM_ = nx; yM_ = ny;
    
    skelft2DFT(0, siteParam_recon, 0, 0, fboSize_, fboSize_, fboSize_);
    
    skelft2DDT(outputFT_recon, 0, xm_, ym_, xM_, yM_);
    //if (sign)
    
    float length = skelft2DMakeBoundary((unsigned char*)outputFT_recon, xm_, ym_, xM_, yM_, siteParam_recon, fboSize_, 0, false);
    if (!length) return NULL;
    
    skelft2DFillHoles((unsigned char*)outputFT_recon, xm_ + 1, ym_ + 1, 1);
    
    skelft2DFT(outputFT_recon, siteParam_recon, xm_, ym_, xM_, yM_, fboSize_);
   
    dt_field(input);
    
    return input;
}