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

// GPU fused NV12 → BGR + resize + tile-place, for the display path.
// Reads packed NV12 from our pool memory and writes a resized BGR tile directly
// into a shared tiled canvas at (dst_x, dst_y), sized cell_w × cell_h.
//
// This is the GPU-composite equivalent of DeepStream's nvmultistreamtiler:
// resize + colour-convert + tile placement all happen on the GPU, so only the
// single small composed canvas crosses PCIe — not N full-resolution frames.
//
// d_canvas : BGR HWC uint8 [canvas_h × canvas_w × 3] device buffer (persistent).
// canvas_w : full canvas width in pixels (row stride = canvas_w * 3 bytes).
void nv12_to_bgr_tile_gpu(
    const uint8_t* d_nv12, int src_w, int src_h,
    uint8_t* d_canvas, int canvas_w,
    int dst_x, int dst_y, int cell_w, int cell_h,
    cudaStream_t stream);
