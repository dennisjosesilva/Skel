#include <iostream>
#include "skeleton_cuda.hpp"
#include "field.h"

int main(int argc, const char *argv[])
{
  FIELD<float> *f = FIELD<float>::read(argv[1]);
  
  f->writePGM("f.pgm");

  int v = initialize_skeletonization(f);
  
  f->threshold(125.f);


  FIELD<float> *skel = computeSkeleton(120, f, 0.002);

  FIELD<float> *out = new FIELD<float>(f->dimX(), f->dimY());
  for (int y = 0; y < skel->dimY(); y++) {
    for (int x = 0; x < skel->dimX(); x++) {      
      if (skel->value(x, y) > 0.0f)
        out->set(x, y, 255);
      else
        out->set(x, y, 0);
    }    
  }

  out->writePGM("output.pgm");
  std::cout << "DONE" << std::endl;   
  return 0;
}