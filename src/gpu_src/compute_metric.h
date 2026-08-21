#ifndef _GPUSHIM_COMPUTE_METRIC_H
#define _GPUSHIM_COMPUTE_METRIC_H
// Backend shim: selects the CUDA or WebGPU implementation of this interface.
// Both expose the same declarations, so main/ needs no other change.
#ifdef USEWEBGPU
    #include "../webgpu_src/compute_metric.h"
#else
    #include "../cuda_src/compute_metric.h"
#endif
#endif
