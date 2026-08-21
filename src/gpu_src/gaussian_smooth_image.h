#ifndef _GPUSHIM_GAUSSIAN_SMOOTH_IMAGE_H
#define _GPUSHIM_GAUSSIAN_SMOOTH_IMAGE_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/gaussian_smooth_image.h"
#else
    #include "../cuda_src/gaussian_smooth_image.h"
#endif
#endif
