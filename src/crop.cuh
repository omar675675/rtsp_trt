#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Convert a full NV12 frame to packed BGR uint8 HWC on GPU.
// d_nv12 — packed NV12: Y plane [src_h × src_w] then UV plane [src_h/2 × src_w].
// d_bgr  — output BGR HWC [src_h × src_w × 3] — caller pre-allocates.
// Launched on 'stream'; caller must sync before reading d_bgr or releasing d_nv12.
void nv12_to_bgr_frame_gpu(
    const uint8_t* d_nv12, int src_w, int src_h,
    uint8_t*       d_bgr,
    cudaStream_t   stream);
