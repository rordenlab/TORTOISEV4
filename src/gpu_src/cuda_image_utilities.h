#ifndef _GPUSHIM_CUDA_IMAGE_UTILITIES_H
#define _GPUSHIM_CUDA_IMAGE_UTILITIES_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/image_utilities_all.h"
#else
    #include "../cuda_src/cuda_image_utilities.h"
#endif
#endif
