#ifndef _WGPU_COMPOSEFIELDS_CXX
#define _WGPU_COMPOSEFIELDS_CXX

#include "compose_fields.h"
#include "webgpu_context.h"
#include "compose_fields.wgsl.h"

struct ComposeParams
{
    int32_t sz[4];
    float   spc[4];
    float   dir[3][4];
};

CUDAIMAGE::Pointer ComposeFields(CUDAIMAGE::Pointer main_field, CUDAIMAGE::Pointer update_field)
{
    CUDAIMAGE::Pointer output = CUDAIMAGE::New();
    output->sz   = main_field->sz;
    output->dir  = main_field->dir;
    output->orig = main_field->orig;
    output->spc  = main_field->spc;
    output->components_per_voxel = main_field->components_per_voxel;
    output->Allocate();

    ComposeParams p{};
    p.sz[0] = main_field->sz.x; p.sz[1] = main_field->sz.y; p.sz[2] = main_field->sz.z;
    p.sz[3] = main_field->components_per_voxel;
    p.spc[0] = main_field->spc.x; p.spc[1] = main_field->spc.y; p.spc[2] = main_field->spc.z;
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            p.dir[r][c] = (float)main_field->dir(r, c);

    wgpu::Buffer params = wgpuctx::CreateUniform(&p, sizeof(p));
    // 32 invocations along x is one coalesced run along a row. The previous 4x4x4 tiling
    // put only 4 in x, so a 32-wide NVIDIA subgroup spanned 8 pitch-separated rows and
    // issued 8 scattered accesses where one would do - the same defect as the CUDA side
    // (see ElementwiseLaunch in cuda_image_utilities.cu), where fixing it measured -39%
    // on NegateImage and -30% on computeFiniteDiffStructs. Shape-agnostic: every
    // elementwise shader indexes by global_invocation_id with an inside() guard.
    // NOT for reductions - reductions.wgsl uses workgroup memory and a fixed geometry.
    const uint32_t wgx = 32, wgy = 4, wgz = 1;
    wgpu::ComputePipeline pipe =
        wgpuctx::Pipeline("compose_fields", kcompose_fieldsWGSL, "main", wgx, wgy, wgz);
    wgpuctx::Dispatch(pipe,
                      {main_field->getFloatdata().buf, update_field->getFloatdata().buf,
                       output->getFloatdata().buf, params},
                      (main_field->sz.x + wgx - 1) / wgx,
                      (main_field->sz.y + wgy - 1) / wgy,
                      (main_field->sz.z + wgz - 1) / wgz);
    return output;
}

#endif
