# Compartmental models parallel GPU-Cuda fitting toolbox

GPU-CUDA toolbox for fitting compartmental models to 4D medical dynamic volumes.

Several models have been already implemented (1TC, 2TC, 2TCr) in the big group of non-linear compartmental models.
More to come (also linear, hopefully, like Patlak and Logan).

The optimization is based on a Maximum-a-Posteriori version of the standard Levemberg-Marquardt algorithm for non linear least squares optimization. 
A couple of local spatial prior are already available (quadratic and Huber). We plan to add more options, like TV and above all some anatomy-related prior, as well.

It uses pyCUDA and cuBLAS python interfaces to parallelize the LM non-linear optimization algorithm.
Next version will probably switch from pyCUDA to CuPY for CUDA interface in python, but we still need to evaluate (suggestions welcome!)

### TODO (more then tide up everything .. sorry for the mess!)
- Add instruction to how to install the library
- Add instruction of how to use the class
- Add some ipython notebooks with examples
