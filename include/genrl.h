#pragma once


const float MYINFINITY = 1.0e7f;
const float eps  	 = 1.0e-6f;
const float eps2 	 = eps*eps;
const int   CONN_UNKNOWN = -1;


struct Coord { 
		int i,j; 
		Coord(int i_,int j_):i(i_),j(j_) {}; 
		Coord() {} 
		int operator==(const Coord& c) const { return i==c.i && j==c.j; } 
	      };

