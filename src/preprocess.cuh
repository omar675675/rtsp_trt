#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct LetterboxInfo {
    float scale;
    int   pad_x, pad_y;
    int   src_w, src_h;
};

// Preprocess one frame from our own CUDA pool memory.
//
// d_nv12  — packed NV12: Y plane [src_h × src_w] bytes then UV plane
//           [src_h/2 × src_w] bytes (stride == src_w, no decoder padding).
//           Allocated by Stream's pool in the primary CUDA context — same
//           context as TRT. No cross-context reads.
// d_dst   — TRT batch slot: float32 RGB CHW [3 × net_h × net_w].
// lb_out  — filled with letterbox geometry for the postprocess inverse transform.
// stream  — CUDA stream (async; caller must cudaStreamSynchronize before TRT).
void preprocess_frame(
    const uint8_t* d_nv12,
    int src_h, int src_w,
    float* d_dst,
    int net_h, int net_w,
    LetterboxInfo& lb_out,
    cudaStream_t stream);
