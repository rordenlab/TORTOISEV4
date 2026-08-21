#ifndef _GPUSHIM_QUADRATIC_TRANSFORM_IMAGE_H
#define _GPUSHIM_QUADRATIC_TRANSFORM_IMAGE_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/quadratic_transform_image.h"
#else
    #include "../cuda_src/quadratic_transform_image.h"
#endif
#endif
