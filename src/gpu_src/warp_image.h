#ifndef _GPUSHIM_WARP_IMAGE_H
#define _GPUSHIM_WARP_IMAGE_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/warp_image.h"
#else
    #include "../cuda_src/warp_image.h"
#endif
#endif
