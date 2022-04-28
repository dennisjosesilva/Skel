#include <iostream>
#include "skeleton_cuda.hpp"
#include "field.h"

int main(int argc, const char *argv[])
{
  FIELD<float> *f = FIELD<float>::read(argv[1]);
  
  FIELD<float> *b = new FIELD<float>{f->dimX(), f->dimY()};
  for (int y = 0; y < f->dimY(); y++) {
    for (int x = 0; x < f->dimX(); x++) {      
      if (f->value(x, y) > 0)
        b->set(x, y, 255);
      else 
        b->set(x, y, 0);
    }
  }
  
  int v = initialize_skeletonization(f);

  FIELD<float> *skel = computeSkeleton(250, b, 5.0);

  for (int y = 0; y < skel->dimY(); y++) {
    for (int x = 0; x < skel->dimX(); x++) {      
      std::cout << skel->value(x, y);
    }
    std::cout << "\n";
  }

  skel->writePGM("output.pgm");
  std::cout << "DONE" << std::endl;   
  return 0;
}