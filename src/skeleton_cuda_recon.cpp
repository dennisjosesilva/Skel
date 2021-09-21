/* skeleton.cpp */

#include <math.h>
#include "fileio.hpp"
#include <cuda_runtime_api.h>
#include "genrl.h"
#include "skelft.h"
#include "skeleton_cuda_recon.hpp"

#define INDEX(i,j) (i)+fboSize_*(j)
float* siteParam_recon, *curr_site, *prev_site, *curr_dt, *prev_dt;

short* outputFT_recon;
float* outputInterp;
//bool* foreground_mask;
int xm_ = 0, ym_ = 0, xM_, yM_, fboSize_, dimX_, dimY_;

void allocateCudaMem_recon(int size) {
    skelft2DInitialization(size);
    cudaMallocHost((void**)&outputFT_recon, size * size * 2 * sizeof(short));
    cudaMallocHost((void**)&outputInterp, size * size * sizeof(float));
    //cudaMallocHost((void**)&foreground_mask, size * size * sizeof(bool));
    cudaMallocHost((void**)&siteParam_recon, size * size * sizeof(float));
    
    cudaMallocHost((void**)&curr_site, size * size * sizeof(float));
    cudaMallocHost((void**)&prev_site, size * size * sizeof(float));
    cudaMallocHost((void**)&curr_dt, size * size * sizeof(float));
    cudaMallocHost((void**)&prev_dt, size * size * sizeof(float));
    
}

void deallocateCudaMem_recon() {
    skelft2DDeinitialization();
    cudaFreeHost(outputFT_recon);
    cudaFreeHost(outputInterp);
    //cudaFreeHost(foreground_mask);
    cudaFreeHost(siteParam_recon);
    cudaFreeHost(curr_site);
    cudaFreeHost(prev_site);
    cudaFreeHost(curr_dt);
    cudaFreeHost(prev_dt);
}

int initialize_skeletonization_recon(int xM, int yM) {
    dimX_ = xM;
    dimY_ = yM;
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


FIELD<float>* getOutput() {
    FIELD<float>* f = new FIELD<float>(dimX_, dimY_);
    for (int i = 0; i < dimX_; ++i) {
        for (int j = 0; j < dimY_; ++j) {
            f -> set(i,j,outputInterp[INDEX(i, j)]);
        }
    }
    return f;
}

FIELD<float> * CUDA_interp(FIELD<float> *curr_l, FIELD<float> *prev_l, FIELD<float> *prev_d, FIELD<float> *curr_d, int curr_bound_value, int prev_bound_value, bool firstL, int lastL)
{
    memset(curr_site, 0, fboSize_ * fboSize_ * sizeof(float));
    memset(prev_site, 0, fboSize_ * fboSize_ * sizeof(float));
    memset(curr_dt, 0, fboSize_ * fboSize_ * sizeof(float));
    memset(prev_dt, 0, fboSize_ * fboSize_ * sizeof(float));

    for (int i = 0; i < dimX_; ++i) {
        for (int j = 0; j < dimY_; ++j) {
            if ((*curr_l)(i, j)) {
                curr_site[INDEX(i, j)] = 1;
            }
            if ((*prev_l)(i, j)) {
                prev_site[INDEX(i, j)] = 1;
            }

            curr_dt[INDEX(i,j)] = (*curr_d)(i, j);
            prev_dt[INDEX(i,j)] = (*prev_d)(i, j);

        }
    }
    
    Interp(outputInterp, curr_site, prev_site, curr_dt, prev_dt, curr_bound_value, prev_bound_value, fboSize_,firstL, lastL);
    FIELD<float> *output = new FIELD<float>(dimX_, dimY_);
    if(lastL) output = getOutput();
    return output;
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