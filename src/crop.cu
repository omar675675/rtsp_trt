#include "crop.cuh"
#include <cuda_runtime.h>

// One thread per output pixel.  BT.601 limited-range YCbCr → BGR uint8.
// Same colour math as nv12_to_bgr_tile_kernel in preprocess.cu; no tile offset.

__global__ void nv12_to_bgr_frame_kernel(
    const uint8_t* __restrict__ d_y,
    const uint8_t* __restrict__ d_uv,
    int src_w, int src_h,
    uint8_t* __restrict__ d_bgr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src_w || y >= src_h) return;

    float Y  = (float)d_y[y * src_w + x] - 16.f;
    int   uv = (y >> 1) * src_w + (x & ~1);
    float U  = (float)d_uv[uv]     - 128.f;
    float V  = (float)d_uv[uv + 1] - 128.f;

    uint8_t b = (uint8_t)fminf(fmaxf(1.164f * Y + 2.018f * U,                0.f), 255.f);
    uint8_t g = (uint8_t)fminf(fmaxf(1.164f * Y - 0.391f * U - 0.813f * V,  0.f), 255.f);
    uint8_t r = (uint8_t)fminf(fmaxf(1.164f * Y              + 1.596f * V,   0.f), 255.f);

    int o = (y * src_w + x) * 3;
    d_bgr[o + 0] = b;
    d_bgr[o + 1] = g;
    d_bgr[o + 2] = r;
}

void nv12_to_bgr_frame_gpu(
    const uint8_t* d_nv12, int src_w, int src_h,
    uint8_t*       d_bgr,
    cudaStream_t   stream)
{
    const uint8_t* d_uv = d_nv12 + (size_t)src_w * src_h;
    dim3 block(16, 16);
    dim3 grid((src_w + 15) / 16, (src_h + 15) / 16);
    nv12_to_bgr_frame_kernel<<<grid, block, 0, stream>>>(
        d_nv12, d_uv, src_w, src_h, d_bgr);
}
