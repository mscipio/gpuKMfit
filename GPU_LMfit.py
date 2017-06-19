#import sys
#sys.path.insert(0, '/local_mount/space/bpet/1/users/michele/python/__functions/')
import os
import pycuda.autoinit 
import pycuda.driver as drv 
import numpy as np 
import pycuda.gpuarray as gpuarray 
from pycuda import compiler 
import skcuda.linalg as culinalg 
import skcuda.misc as cumisc 
import skcuda.cublas as cublas 
import timeit
from string import Template 
from pylab import *
import IFmodels as ifm

culinalg.init() 
import string 
import time 

start = drv.Event()
end = drv.Event()

handle = cumisc._global_cublas_handle
dev = cumisc.get_current_device()
print "GPU:   ", dev.name()
print "Memory: %1.2f Gb"%(dev.total_memory()/1024.0**3)

def bptrs(a):
    """
    Pointer array when input represents a batch of matrices or vectors.
    Taken from scikits.cuda tests/test_cublas.py
    """
    return gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)

class KineticModelFitterLM(): 
    def __init__(self, shape=(64,64,64), Nt=24):
        self.Nt             = Nt                                        # number of time points 
        self.N              = shape[0]*shape[1]*shape[2]                # number of (image) elements 
        self.D              = 5                                         # number of parameters of the kinetic model 
        self.shape          = shape                                     # (image) shape 
        self.shape_y        = ( shape[0], shape[1], shape[2], self.Nt)   # shape of observation data 
        self.shape_par      = (shape[0], shape[1], shape[2], self.D)    # shape of 4D parameters matrix 
        self.eps            = np.float32(1e-16)
        self.ll             = np.float32(0.0)                           # weight lambda (LM algorithm)
        self.bb             = np.float32(0.0)                           # weight beta (MRF prior importance)
	self.gamma          = np.float32(0.0)                           # weight gamma (Thikonov term in MRF prior)
	self.func_path      = os.path.dirname(os.path.abspath(__file__))
	self.kernel_models  = open(self.func_path+'/cuda_kernels/models.cu').read()
	self.kernel_priors  = open(self.func_path+'/cuda_kernels/priors.cu').read()
        self._init_gpu_()                                               # allocate GPU memory 

    def _init_gpu_(self): 
        """Alloc memory on GPU"""

        self.y_gpu     = gpuarray.empty( (self.N, self.Nt) , np.float32 )         # observations 
        self.f_gpu     = gpuarray.zeros( (self.N, self.Nt) , np.float32 )         # model evaluation 
	self.mask_gpu  = gpuarray.zeros( self.shape , np.float32 )

        self.par_gpu   = gpuarray.empty( (self.N, self.D, 1) , np.float32 )       # parameters 
	self.par_out   = np.zeros( (self.N, self.D) ) 
        self.dk_gpu    = gpuarray.empty( 1, np.float32 ) 

        self.g_gpu     = gpuarray.empty( (self.N, self.D, 1), np.float32 )        # gradient 
        self.g_arr     = bptrs(self.g_gpu)
	self.prior_gpu = gpuarray.empty( (self.N, self.D, 1), np.float32 )        # MRF prior (same shape as self.g_gpu)
        self.delta_gpu = gpuarray.empty( (self.N, self.D, 1), np.float32 )        # gradient 
        self.delta_arr = bptrs(self.delta_gpu)
        
	self.Diff_gpu  = gpuarray.empty( (self.N, self.Nt, 1), np.float32 )       # self.y_gpu - self.f_gpu
        self.Diff_arr  = bptrs(self.Diff_gpu) 
        self.J_gpu     = gpuarray.empty( (self.N, self.Nt, self.D), np.float32 )  # Jacobian
        self.J_arr     = bptrs(self.J_gpu) 
        self.H_gpu     = gpuarray.empty( (self.N, self.D, self.D), np.float32 )   # (approximate) Hessian
        self.H_arr     = bptrs(self.H_gpu) 
        self.Hinv_gpu  = gpuarray.empty( (self.N, self.D, self.D), np.float32 )   # (approximate) inverse Hessian 
        self.Hinv_arr  = bptrs(self.Hinv_gpu) 
        I              = np.tile(self.ll*np.eye(self.D),[self.N,1,1])
        self.I_gpu     = gpuarray.to_gpu(I.astype('float32'))

        self._h        = cumisc._global_cublas_handle                              # GPU handle
        self._info_gpu = gpuarray.zeros( self.N, np.int32)                         # info matrix utilized by batched cublas 
        self._p_gpu    = gpuarray.empty( (self.N, self.D), np.int32)               # info matrix utilized by batched cublas 
      
    def step(self, verbose=True, smoothing=0.0):

        start = timeit.default_timer()

        # 1- compute model & jacobian
	grid_dim = np.int(np.ceil((self.N+self.B-1)/self.B))
        self.parallel_analytic_models(block_dim=(self.B,1,1), grid_dim=(grid_dim,1,1))
        elapsed_sim = timeit.default_timer() - start
        # 1a- Remove sparse voxels from parametric maps 
        #   - Transfer the change to self.f_gpu (clustering meaningful voxel, 
        #                                        in the hypothesis that they 
        #                                        should be contiguos in space)
        
        # 2- evaluate error - keep error history 
        self.Diff_gpu[:,:,0] = self.y_gpu - self.f_gpu
	
	"""figure(figsize=[14,6])
	for par in range(5):
	  subplot(2,3,par+1); plot(self.time_gpu.get(),self.J_gpu.get()[self.N-1,:,par])"""


        # 3- compute approximate Hessians
        alpha = np.float32(1.0)
        beta  = np.float32(0.0)

        cublas.cublasSgemmBatched(handle, 'n','t',
                                  self.D, self.D, self.Nt, alpha,
                                  self.J_arr.gpudata, self.D,
                                  self.J_arr.gpudata, self.D,
                                  beta, self.H_arr.gpudata, self.D, self.N)
	
        self.H_gpu  += self.I_gpu
        
        # 4- compute approximate inverse Hessians 
        cublas.cublasSgetrfBatched(self._h, self.D, self.H_arr.gpudata, self.D, self._p_gpu.gpudata, self._info_gpu.gpudata, self.N)
        cublas.cublasSgetriBatched(self._h, self.D, self.H_arr.gpudata, self.D, self._p_gpu.gpudata, self.Hinv_arr.gpudata, self.D, self._info_gpu.gpudata, self.N)

        # 5- compute gradient 
        # use batched matrix vector multiplication to compute  self.g_gpu = inner( self.J_gpu.T, self.y_gpu-self.f_gpu )
        cublas.cublasSgemmBatched(handle, 'n','t',
                                  1, self.D, self.Nt, alpha,
                                  self.Diff_arr.gpudata, 1,
                                  self.J_arr.gpudata, self.D,
                                  beta, self.g_arr.gpudata, 1, self.N)

	# 6- compute derivative of smoothing gaussian MRF log prior
	self.compute_prior(block_dim=(self.B,1,1,), grid_dim=(np.int(np.ceil((self.N+self.B-1)/self.B)),1,1))
	self.g_gpu  += self.prior_gpu

        # 7- compute preconditioned gradient 
        # use batched matrix vector multiplication to compute  self.g_gpu = inner( self.Hinv_gpu, self.g_gpu )
        cublas.cublasSgemmBatched(handle, 'n','n',
                                  1, self.D, self.D, alpha,
                                  self.g_arr.gpudata, 1,
                                  self.Hinv_arr.gpudata, self.D,
                                  beta, self.delta_arr.gpudata, 1, self.N)

        # 8- update parameters 
        #self.par_gpu += self.delta_gpu 
	# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO: do this on GPU 
	par_new = self.par_gpu.get() + self.delta_gpu.get() 
	par_new[par_new<=0] = self.eps
	self.par_gpu = gpuarray.to_gpu(np.asarray(par_new, dtype=np.float32))
	# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO: do this on GPU 

        if verbose:
	    elapsed_iter = timeit.default_timer() - start
            print '-------------------------------------------'
            print 'simulation time: '+ str(elapsed_sim)+' sec'
            #print "par_GPU_new"+str(self.par_gpu[0,:,0])
            print 'iteration time: '+ str(elapsed_iter)+' sec'
            print '-------------------------------------------'
	else:
	    #print "..."
	    pass
             
    def parallel_analytic_models(self, block_dim=None, grid_dim=None):       
        
        kernel_code_template = Template(self.kernel_models)
        
        if block_dim is None or grid_dim is None: 
            block_dim, grid_dim = cumisc.select_block_grid_sizes(pycuda.autoinit.device, self.shape)

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code_template.substitute(max_threads_per_block=block_dim[0],
                                              max_blocks_per_grid=grid_dim[0],
                                              W=self.shape[1], L=self.shape[2],T=self.Nt, D=self.D, N=self.N))

        # get the kernel function from the compiled module
        self.model_fun = mod.get_function(self.model)          

	# TODO fill output array with zeros here instead than in the kernel_code_template
	# ES. self.f_gpu.fill(0

        # call the kernel on the card
        self.model_fun(
                # inputs
                self.par_gpu, self.IFparams_gpu, self.IF_gpu, self.time_gpu, 
                # output
                self.f_gpu, self.J_gpu,
                # params
                self.dk_gpu, self.mask_gpu,
                #self.s_gpu, self.d_gpu, self.p_gpu, self.Ahat_gpu, 
                # grid of multiple blocks
                grid = grid_dim,
                # block of multiple threads
                block = block_dim, 
                )           
    
    def compute_prior(self, block_dim=None, grid_dim=None):
        kernel_code_template = Template(self.kernel_priors)
        
        if block_dim is None or grid_dim is None: 
            block_dim, grid_dim = cumisc.select_block_grid_sizes(pycuda.autoinit.device, self.shape)

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code_template.substitute(max_threads_per_block=block_dim[0],
                                              max_blocks_per_grid=grid_dim[0],
                                              W=self.shape[1], L=self.shape[2], D=self.D, N=self.N))

        # get the kernel function from the compiled module
        self.compute_gradient_log_prior = mod.get_function(self.prior)

        # call the kernel on the card
        self.compute_gradient_log_prior(
            # inputs
            self.par_gpu,
            # output
            self.prior_gpu, 
            # par
            self.bb, self.gamma, self.thresh, self.mask_gpu,
            # grid of multiple blocks
            grid = grid_dim,
            # block of multiple threads
            block = block_dim, 
            )

        
    def fit(self, iterations=20, B = 128, verbose = True, saveCost = True, smoothing=0.0): 
        """Fit parameters."""
	#print "CHOOSEN MODEL: "+ self.model
	if saveCost:
	  self.CostFun = np.zeros((self.N,iterations))
        self.B = B
        start = timeit.default_timer()
        for it in range(iterations): 
	    #print self.f_gpu.get()[:,:]
            self.step(verbose,smoothing)
            if saveCost:
                self.CostFun[:,it] = norm(self.Diff_gpu.get()[:,:,0],2,1)
            #figure(figsize=[8,4])
            #plot(mean(time,0),self.y_gpu.get().reshape([128,128,47,24])[120,100,20,:], 'r*')
            #plot(mean(time,0),self.f_gpu.get().reshape([128,128,47,24])[120,100,20,:], 'b-')
        elapsed = timeit.default_timer() - start
        #print '======================= TOTAL fitting time: '+ str(elapsed) +"sec ======================="
        if saveCost:
            figure(figsize=[8,4])
            plot(self.CostFun[0,:])
            suptitle("Cost Function", fontsize=20, fontweight='bold')
            xlabel("iteration number")

    def k2aux(self,vB=0.01, K1=0.1, k2=0.1, k3=0.01, k4=0.01):
	if (self.model == "bicompartment_3expIF_4k") or (self.model == "bicompartment_2expIF_4k"):
	  s  = (k2+k3+k4)
	  d  = np.abs(np.sqrt(s**2 - 4*k2*k4));
	  p2 = (s + d) / 2                #L1
	  p4 = (s - d) / 2                #L2
	  p1 = (K1 * ( p2 - k3 - k4)) / d #B1
	  p3 = (K1 * (-p4 + k3 + k4)) / d #B2
	  self.par_gpu[:,:,0]  = gpuarray.to_gpu(np.asarray([[vB, p1,p2,p3,p4]]*self.N, dtype=np.float32))
	elif (self.model == "bicompartment_3expIF_3k") or (self.model == "bicompartment_2expIF_3k"):
	  p2 = k2 + k3        #L1
	  p4 = 0              #L2
	  p1 = K1 *  k2 / p2  #B1
	  p3 = K1 *  k3 / p2  #B2 
	  self.par_gpu[:,:,0]  = gpuarray.to_gpu(np.asarray([[vB, p1,p2,p3,p4]]*self.N, dtype=np.float32))
	elif (self.model == "monocompartment_3expIF") or (self.model == "monocompartment_2expIF"):
	  p2 = k2        #L1
	  p4 = 0         #L2
	  p1 = K1        #B1
	  p3 = 0         #B2 
	  self.par_gpu[:,:,0]  = gpuarray.to_gpu(np.asarray([[vB, p1,p2,p3,p4]]*self.N, dtype=np.float32))

    def aux2k(self):
	if (self.model == "bicompartment_3expIF_4k") or (self.model == "bicompartment_2expIF_4k"):
	  n  = self.par_gpu.get()[:,1,0]*self.par_gpu.get()[:,2,0] + self.par_gpu.get()[:,3,0]*self.par_gpu.get()[:,4,0]
	  d  = self.par_gpu.get()[:,1,0] + self.par_gpu.get()[:,3,0]
	  self.par_out[:,0] = self.par_gpu.get()[:,0,0]
	  self.par_out[:,1] = d
	  self.par_out[:,2] = n / d
	  self.par_out[:,3] = (self.par_gpu.get()[:,1,0] * self.par_gpu.get()[:,3,0] * (self.par_gpu.get()[:,2,0] - self.par_gpu.get()[:,4,0])**2 ) / (d * n)
	  self.par_out[:,4] = (self.par_gpu.get()[:,2,0] * self.par_gpu.get()[:,4,0] * (self.par_gpu.get()[:,1,0] + self.par_gpu.get()[:,3,0])) / n
	elif (self.model == "bicompartment_3expIF_3k")  or (self.model == "bicompartment_2expIF_3k"):
	  self.par_out[:,0] = self.par_gpu.get()[:,0,0]
	  self.par_out[:,1] = self.par_gpu.get()[:,1,0] + self.par_gpu.get()[:,3,0]
	  self.par_out[:,2] = self.par_gpu.get()[:,1,0] * self.par_gpu.get()[:,2,0] / self.par_out[:,1]
	  self.par_out[:,3] = self.par_gpu.get()[:,2,0] * self.par_gpu.get()[:,3,0] / self.par_out[:,1]
	  self.par_out[:,4] = 0
	elif (self.model == "monocompartment_3expIF") or (self.model == "monocompartment_2expIF"): #monocompartmental models
	  self.par_out[:,0] = self.par_gpu.get()[:,0,0] #vb
	  self.par_out[:,1] = self.par_gpu.get()[:,1,0] #k1
	  self.par_out[:,2] = self.par_gpu.get()[:,2,0] #k2
	  self.par_out[:,3] = 0
	  self.par_out[:,4] = 0
    
    def get_parameters(self):
        """Get the fitted parameters of the kinetic model. """
        # TODO: reshape with self.shape_par
	self.aux2k()
        return self.par_out 
    
    def get_jacobian(self):
        """Get the fitted parameters of the kinetic model. """
        return self.J_gpu.get()
    
    def get_fit_result(self):
        """Get the fitted parameters of the kinetic model. """
        # TODO: reshape with self.shape_y
        return self.f_gpu.get()

    def initialize(self, scanTime, observations, IF, IFparams=[],IF_model=2, model="bicompartment_3expIF_4k", prior="gaussian_MRF_prior", vB=0.01, K1=0.1, k2=0.1, k3=0.01, k4=0.01, dk=np.log(2)/109.8, l=1e0, b=[0.1e0,0.1e0,0.1e0,0.1e0,0.1e0], g=1e0, t=[1e0,1e1,1e1,1e0,0], mask = None): 
        """Run all the 'set' functions in the right order."""
        self.Nt = np.asarray(scanTime).shape[1]
        self.N  = np.asarray(observations).shape[0]
        self.ll = float32(l)  
	if len(b) == 1:
	  self.bb = gpuarray.to_gpu(np.asarray([b]*self.D, dtype=np.float32))
	elif len(b) == self.D:
	  self.bb = gpuarray.to_gpu(np.asarray(b, dtype=np.float32))
	else:
	  self.bb = gpuarray.to_gpu(np.asarray([[0.1e0,0.1e0,0.1e0,0.1e0,0.1e0]]*self.D, dtype=np.float32))
	  print "Wrong size for beta array (prior weight)"
	self.gamma = float32(g)
	if len(t) == 1:
	  self.thresh = gpuarray.to_gpu(np.asarray([t]*self.D, dtype=np.float32))
	elif len(t) == self.D:
	  self.thresh = gpuarray.to_gpu(np.asarray(t, dtype=np.float32))
	else:
	  self.thresh = gpuarray.to_gpu(np.asarray([[1e0,1e1,1e1,1e0,0]]*self.D, dtype=np.float32))
	  print "Wrong size for thresh array (sparsity threshold)"
	self.model = model 
	self.prior = prior
        self._init_gpu_()
        self.set_time(scanTime)
        self.set_observations(observations) 
        self.set_input_function(IF, scanTime, IFparams,IF_model)
        self.set_input_params(vB, K1, k2, k3, k4, dk)
	if mask is  None:
	  self.mask_gpu = gpuarray.to_gpu(np.asarray(np.ones(self.shape), dtype=np.float32))
	else:
	  self.mask_gpu = gpuarray.to_gpu(np.asarray(mask, dtype=np.float32))

    def set_observations(self, observations):
        """Set noisy observations to be fit."""
        # TODO: check if a reshape is needed
        self.y_gpu  = gpuarray.to_gpu(np.asarray(observations, dtype=np.float32))
        
    def set_input_function(self, IF, scanTime, IFparams, IF_model=2):
        """Import input function and IF parameters for the choosen model.
        You can either input those parameters or ask to fit the choosen model to the experimental data"""
	if (IFparams == []) or (IFparams==None) :
	  params = ifm.fit(IF,scanTime, model=IF_model)
	  IFfit  = ifm.simulate(scanTime, params, model=IF_model)
	  self.IFparams_gpu    = gpuarray.to_gpu(np.asarray(params, dtype=np.float32))
	  self.IF_gpu          = gpuarray.to_gpu(np.asarray(IFfit,  dtype=np.float32))
	  
	  print "IF PARAMETERS "
	  print "----------------------------------------------------------------"
	  print "delay: "+ str(self.IFparams_gpu.get()[0])
	  if IF_model==2 or IF_model == 3:
	    print "amplitudes: "+ str(self.IFparams_gpu.get()[1:4])
	    print "time constants: "+ str(self.IFparams_gpu.get()[4:])
	  elif IF_model==4 or IF_model==6:
	    print "amplitudes: "+ str(self.IFparams_gpu.get()[1:3])
	    print "time constants: "+ str(self.IFparams_gpu.get()[3:])
	  elif IF_model == 5:
	    print "amplitudes: "+ str(self.IFparams_gpu.get()[1])
	    print "time constants: "+ str(self.IFparams_gpu.get()[2:])
	  print "----------------------------------------------------------------"

	  plot(self.time_gpu.get(),IF,'r*-' )
	  plot(self.time_gpu.get(),IFfit,'b-' )
	  suptitle("Input function model", fontsize="18", fontweight="bold")
	  show()
        else:
	  self.IFparams_gpu    = gpuarray.to_gpu(np.asarray(IFparams, dtype=np.float32))
	  self.IF_gpu          = gpuarray.to_gpu(np.asarray(IF,       dtype=np.float32))
        
    def set_time(self, scanTime): 
	if np.asarray(scanTime).max()>180:
	   scanTime = np.asarray(scanTime)/60; #time has to be in minutes
        self.time_gpu       = gpuarray.to_gpu(np.asarray(np.mean(scanTime,0), dtype=np.float32))

    def set_input_params(self, vB=0.01, K1=0.1, k2=0.1, k3=0.01, k4=0.01, dk=np.log(2)/109.8): 
        """Initialize the parameters of the kinetic model. """
        # TODO: allow to input a volume of initial parameters (also for simulation purpose)
	self.k2aux(vB+self.eps, K1+self.eps, k2+self.eps, k3+self.eps, k4+self.eps) # self.par_gpu is defined in the auxiliary variables domain, not in the 'k' domain
        self.dk_gpu   = gpuarray.to_gpu(np.asarray(dk, dtype=np.float32))   

    def __del__(self):
        pass
        # free GPU memory 
    
