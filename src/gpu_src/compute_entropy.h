#ifndef _GPUSHIM_COMPUTE_ENTROPY_H
#define _GPUSHIM_COMPUTE_ENTROPY_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/compute_entropy.h"
#else
    #include "../cuda_src/compute_entropy.h"
#endif
#endif
