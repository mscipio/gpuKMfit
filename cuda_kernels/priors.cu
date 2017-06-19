#include <stdio.h>
// Macro for converting subscripts to linear index:
#define INDEX_PARAM(i, p) i*${D}+p
#define INDEX_4D(x, y, z, p) x*${W}*${L}*${D}+y*${L}*${D}+z*${D}+p
#define INDEX_MASK(x, y, z) x*${W}*${L}+y*${L}+z
    
__global__ void gaussian_MRF_prior(float *k4D, float *prior, float *beta, float gamma,float *threshold, float *mask)
{
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Convert the linear index to 4D subscripts:
    unsigned int x = idx/(${W}*${L});
    unsigned int y = (idx%(${W}*${L}))/${L};
    unsigned int z = (idx%(${W}*${L}))%${L};
    
    float sum; //[${D}];
    int m;
    int temp;
    
    // Use the 4D subscripts to access the arrays:
    if (idx < ${N}) {    
    
        for (uint dd=0; dd<${D}; dd++) {
            prior[INDEX_PARAM(idx,dd)] = 0;
	    
	    if ((idx*mask[INDEX_MASK(x, y, z)]!= 0) || (idx == 0 && mask[INDEX_MASK(x, y, z)]!= 0)){
	      //prior[INDEX_PARAM(idx,dd)] = 10;
	      		
	      sum = 0;
	      m = 0;
	      temp = z-1;
	      if (temp >= 0)  {
		  sum -= k4D[INDEX_4D(x, y, temp, dd)];   //k4D[INDEX_4D(x, y, z-1, dd)];
		  m++;
	      }
	      temp = z+1;
	      if (temp < ${L}){
		  sum -= k4D[INDEX_4D(x, y, temp, dd)];   //k4D[INDEX_4D(x, y, z+1, dd)];
		  m++;
	      }
	      temp = x-1;
	      if (temp >= 0)  {
		  sum -= k4D[INDEX_4D(temp, y, z, dd)];   //k4D[INDEX_4D(x-1, y, z, dd)];
		  m++;
	      }
	      temp = x+1;
	      if (temp < ${W}){
		  sum -= k4D[INDEX_4D(temp, y, z, dd)];   //k4D[INDEX_4D(x+1, y, z, dd)];
		  m++;
	      }
	      temp = y-1;
	      if (temp >= 0)  {
		  sum -= k4D[INDEX_4D(x, temp, z, dd)];   //k4D[INDEX_4D(x, y-1, z, dd)];
		  m++;
	      }
	      temp = y+1;
	      if (temp < ${W}){
		  sum -= k4D[INDEX_4D(x, temp, z, dd)];   //k4D[INDEX_4D(x, y+1, z, dd)];
		  m++;
	      }
	      sum += (m+gamma)*k4D[INDEX_4D(x, y, z, dd)] ;
	      prior[INDEX_PARAM(idx,dd)] -= beta[dd]*sum;
	    }

        }
    }
}

__global__ void sparsity_prior(float *k4D, float *prior, float *beta, float gamma, float *threshold, float *mask)
{
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Convert the linear index to 4D subscripts:
    unsigned int x = idx/(${W}*${L});
    unsigned int y = (idx%(${W}*${L}))/${L};
    unsigned int z = (idx%(${W}*${L}))%${L};
    
    float sum; //[${D}];
    int temp;
    float diff;
    float thresh;
    
    // Use the 4D subscripts to access the arrays:
    if (idx < ${N}) {    
    
        for (uint dd=0; dd<${D}; dd++) {
            prior[INDEX_PARAM(idx,dd)] = 0;
	    
	    if ((idx*mask[INDEX_MASK(x, y, z)]!= 0) || (idx == 0 && mask[INDEX_MASK(x, y, z)]!= 0)){

	      sum = 0;
	      diff = 0;
	      thresh = threshold[dd];
	      
	      temp = z-1;
	      if (temp >= 0)  {
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(x, y, temp, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(x, y, temp, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(x, y, temp, dd)] ;}
	      }
	      temp = z+1;
	      if (temp < ${L}){
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(x, y, temp, dd)];   //k4D[INDEX_4D(x, y, z+1, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(x, y, temp, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(x, y, temp, dd)] ;}
	      }
	      temp = x-1;
	      if (temp >= 0)  {
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(temp, y, z, dd)];   //k4D[INDEX_4D(x-1, y, z, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(temp, y, z, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(temp, y, z, dd)] ;}
	      }
	      temp = x+1;
	      if (temp < ${W}){
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(temp, y, z, dd)];   //k4D[INDEX_4D(x+1, y, z, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(temp, y, z, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(temp, y, z, dd)] ;}
	      }
	      temp = y-1;
	      if (temp >= 0)  {
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(x, temp, z, dd)];   //k4D[INDEX_4D(x, y-1, z, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(x, temp, z, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(x, temp, z, dd)] ;}
	      }
	      temp = y+1;
	      if (temp < ${W}){
		diff = k4D[INDEX_4D(x, y, z, dd)] - k4D[INDEX_4D(x, temp, z, dd)];   //k4D[INDEX_4D(x, y+1, z, dd)];
		if (-thresh < diff < thresh){
		  sum += diff;}
		else if (diff >= thresh){
		  sum += thresh - k4D[INDEX_4D(x, temp, z, dd)] ;}
		else if (diff <= -thresh){
		  sum -= thresh - k4D[INDEX_4D(x, temp, z, dd)] ;}
	      }
	      sum += gamma  *k4D[INDEX_4D(x, y, z, dd)];
	      prior[INDEX_PARAM(idx,dd)] -= beta[dd]*sum;
	    }
        }
    }
}