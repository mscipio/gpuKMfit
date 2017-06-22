#include <stdio.h>
// Macro for converting subscripts to linear index:
#define INDEX_VOL_TIME(i, t) i*${T} +t
#define INDEX_JAC_TIME(i, t, p) i*${T} *${D} +t*${D} +p
#define INDEX_PARAM(i, p) i*${D} +p
#define INDEX_MASK(x, y, z) x*${W} *${L} +y*${L} +z

/*******************************************************************************************************************************
                          MODEL FUNCTIONS DECLARATION (see the end of this file for the body of the functions)
*******************************************************************************************************************************/
__device__ void bicomp_2expIF_noDecay(unsigned int idx, float *aux_par, float *inputfuns, float *IF, float *times, float *func, float *jac, float *mask);

/*******************************************************************************************************************************
                                                      DCE-MRI
*******************************************************************************************************************************/

// MONOOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4)
__global__ void monocompartment_2expIF(float *aux_par, float *inputfun, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{
	// Obtain the linear index corresponding to the current thread:
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	//*** Uncomment the lines below to use shared memory
	__shared__ float times[${T}];
	__shared__ float inputfuns[5];
	if (threadIdx.x < ${T}) {
		times[threadIdx.x] = time[threadIdx.x];
		if (threadIdx.x < 5)
			inputfuns[threadIdx.x] = inputfun[threadIdx.x];
	}
	__syncthreads();
	//*** Comment the lines above and uncomment below to disable shared memory
	//float *times = time;
	//float *inputfuns = inputfun;
	//***

	bicomp_2expIF_noDecay(idx, aux_par, inputfuns, IF, times, func, jac, mask);
	//__syncthreads();

	// deactivate the jacobian for the 4th kinetic constant we don't want to update
	for (uint tt=0; tt<${T}; ++tt) {
		jac[INDEX_JAC_TIME(idx,tt,3)] = 0;
		jac[INDEX_JAC_TIME(idx,tt,4)] = 0;
	}
	//__syncthreads();
}


/*******************************************************************************************************************************
                                                 COMPARTMENTAL MODELS IMPLEMENTATION
*******************************************************************************************************************************/
// ANALYTIC FORMULATION OF A BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4) -- NO DECAY CORRECTION FOR DCE-MRI
__device__ void bicomp_2expIF_noDecay(unsigned int idx, float *aux_par, float *inputfuns, float *IF, float *times, float *func, float *jac, float *mask)
{
	float delta0;
	float delta;
	float p[4];
	float Ahat[2];
	float Abar[2];
	float sum[${T}];
	float TAC[${T}];
	float Jb[${T}];
	float Jl[${T}];

	unsigned int x = idx/(${W} *${L});
	unsigned int y = (idx%(${W} *${L}))/${L};
	unsigned int z = (idx%(${W} *${L}))%${L};
	// __syncthreads();

	// Compute output of bicompartmental model and Jacobian using analytical expression.
	if (idx < ${N}) {
		/* Auxiliary parameters

		   p[1] = k[INDEX_PARAM(idx,2)] + k[INDEX_PARAM(idx,3)];   //L1
		   p[3] = 0;   //L2
		   p[0] = k[INDEX_PARAM(idx,1)] *  k[INDEX_PARAM(idx,2)] / p[1];  //B1
		   p[2] = k[INDEX_PARAM(idx,1)] *  k[INDEX_PARAM(idx,3)] / p[1];  //B2 */
		p[1] = aux_par[INDEX_PARAM(idx,2)];
		p[3] = aux_par[INDEX_PARAM(idx,4)];
		p[0] = aux_par[INDEX_PARAM(idx,1)];
		p[2] = aux_par[INDEX_PARAM(idx,3)];
		Abar[0] = -inputfuns[2];
		Abar[1] =  inputfuns[2];

		for (uint tt=0; tt<${T}; ++tt) { // reset the values of TAC and JAC for current voxel/thread
			func[INDEX_VOL_TIME(idx,tt)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,0)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,1)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,2)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,3)] = 0;
			jac[INDEX_JAC_TIME(idx,tt,4)] = 0;
			TAC[tt] = 0;
		}
		//__syncthreads();
		if ((idx*mask[INDEX_MASK(x, y, z)]!= 0) || (idx == 0 && mask[INDEX_MASK(x, y, z)]!= 0)) {
			for (uint ii=0; ii<=2; ii+=2) { //i = 1:2:4 % 2 compartiments
				delta0  = 1.0f / (p[ii+1] + inputfuns[3]);
				Ahat[0] = -inputfuns[2]-(inputfuns[1]/delta0);
				Ahat[1] = inputfuns[2];

				for (uint tt=0; tt<${T}; ++tt) { // reset temporary variables for i-th compartment
					sum[tt]=0;
					Jb[tt] =0;
					Jl[tt] =0;
				}
				for (uint jj=0; jj<2; ++jj) {
					delta  = 0;

					for (uint tt=0; tt<${T}; ++tt) {
						if (times[tt]>=inputfuns[0]) {
							sum[tt] += idx;
							Jb[tt]  += idx;
							Jl[tt]  += idx;
						}
					}
				}

				for (uint tt=0; tt<${T}; ++tt) {
					if (times[tt]>=inputfuns[0]) {
						TAC[tt] += sum[tt];
						jac[INDEX_JAC_TIME(idx,tt,ii+1)] = Jb[tt];
						jac[INDEX_JAC_TIME(idx,tt,ii+2)] = Jl[tt];
					}
				}

			}
			//__syncthreads();
			for (uint tt=0; tt<${T}; ++tt) {
				jac[INDEX_JAC_TIME(idx,tt,0)] = idx;
				//TAC[tt]  = ((1-aux_par[INDEX_PARAM(idx,0)]) * TAC[tt]) + (aux_par[INDEX_PARAM(idx,0)] * IF[tt]);
				func[INDEX_VOL_TIME(idx,tt)] += TAC[tt];
			}
			//__syncthreads();
		}
	}
}
