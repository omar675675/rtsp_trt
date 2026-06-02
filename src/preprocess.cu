#include "preprocess.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// One thread per output pixel.
// Input: packed NV12 in our own CUDA memory (stride == src_w, no padding).
// Output: RGB float32 CHW in the TRT batch input slot.
// BT.601 limited-range YCbCr → RGB normalised to [0, 1].

__global__ void letterbox_nv12_kernel(
    const uint8_t* __restrict__ d_y,    // Y  plane: src_h × src_w (packed)
    const uint8_t* __restrict__ d_uv,   // UV plane: src_h/2 × src_w (packed, interleaved Cb/Cr)
    float*         __restrict__ d_dst,
    int src_h, int src_w,
    int net_h, int net_w,
    float scale, int pad_x, int pad_y)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= net_w || oy >= net_h) return;

    float sx = (ox - pad_x) / scale;
    float sy = (oy - pad_y) / scale;

    float r, g, b;
    if (sx < 0.f || sy < 0.f || sx >= src_w || sy >= src_h) {
        r = g = b = 114.f / 255.f;   // YOLO letterbox grey
    } else {
        int ix = min(__float2int_rd(sx), src_w - 1);
        int iy = min(__float2int_rd(sy), src_h - 1);

        float Y  = (float)d_y[iy * src_w + ix];
        int   uv = (iy >> 1) * src_w + (ix & ~1);
        float U  = (float)d_uv[uv];        // Cb
        float V  = (float)d_uv[uv + 1];    // Cr

        // BT.601 limited-range (16-235 Y, 16-240 UV) → RGB [0-255]
        float Yf = Y - 16.f, Uf = U - 128.f, Vf = V - 128.f;
        r = __saturatef((1.164f * Yf              + 1.596f * Vf) * (1.f / 255.f));
        g = __saturatef((1.164f * Yf - 0.391f * Uf - 0.813f * Vf) * (1.f / 255.f));
        b = __saturatef((1.164f * Yf + 2.018f * Uf)               * (1.f / 255.f));
    }

    const int stride = net_h * net_w;
    const int idx    = oy * net_w + ox;
    d_dst[0 * stride + idx] = r;
    d_dst[1 * stride + idx] = g;
    d_dst[2 * stride + idx] = b;
}

// ── NV12→BGR + resize + tile kernel (display compositor) ──────────────────────
// One thread per output pixel of the destination cell.  Nearest-neighbour
// downscale from the source frame + BT.601 YCbCr→BGR, written straight into the
// shared canvas at the tile offset.  GPU equivalent of nvmultistreamtiler.

__global__ void nv12_to_bgr_tile_kernel(
    const uint8_t* __restrict__ d_y,   // Y  plane: src_h × src_w
    const uint8_t* __restrict__ d_uv,  // UV plane: src_h/2 × src_w
    int src_w, int src_h,
    uint8_t* __restrict__ d_canvas, int canvas_w,
    int dst_x, int dst_y, int cell_w, int cell_h)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cell_w || cy >= cell_h) return;

    // Nearest-neighbour map: cell pixel → source pixel
    int sx = cx * src_w / cell_w; if (sx >= src_w) sx = src_w - 1;
    int sy = cy * src_h / cell_h; if (sy >= src_h) sy = src_h - 1;

    float Y  = (float)d_y[sy * src_w + sx] - 16.f;
    int   uv = (sy >> 1) * src_w + (sx & ~1);
    float U  = (float)d_uv[uv]     - 128.f;  // Cb
    float V  = (float)d_uv[uv + 1] - 128.f;  // Cr

    uint8_t b = (uint8_t)fminf(fmaxf(1.164f * Y + 2.018f * U             , 0.f), 255.f);
    uint8_t g = (uint8_t)fminf(fmaxf(1.164f * Y - 0.391f * U - 0.813f * V, 0.f), 255.f);
    uint8_t r = (uint8_t)fminf(fmaxf(1.164f * Y              + 1.596f * V, 0.f), 255.f);

    int px = dst_x + cx, py = dst_y + cy;
    int o  = (py * canvas_w + px) * 3;
    d_canvas[o + 0] = b;
    d_canvas[o + 1] = g;
    d_canvas[o + 2] = r;
}

void nv12_to_bgr_tile_gpu(
    const uint8_t* d_nv12, int src_w, int src_h,
    uint8_t* d_canvas, int canvas_w,
    int dst_x, int dst_y, int cell_w, int cell_h,
    cudaStream_t stream)
{
    const uint8_t* d_uv = d_nv12 + (size_t)src_w * src_h;
    dim3 block(16, 16);
    dim3 grid((cell_w + 15) / 16, (cell_h + 15) / 16);
    nv12_to_bgr_tile_kernel<<<grid, block, 0, stream>>>(
        d_nv12, d_uv, src_w, src_h,
        d_canvas, canvas_w, dst_x, dst_y, cell_w, cell_h);
}

// ── NV12→float RGB CHW letterbox (TRT input) ─────────────────────────────────

void preprocess_frame(
    const uint8_t* d_nv12,
    int src_h, int src_w,
    float* d_dst,
    int net_h, int net_w,
    LetterboxInfo& lb_out,
    cudaStream_t stream)
{
    float scale  = std::min((float)net_w / src_w, (float)net_h / src_h);
    int scaled_w = (int)std::round(src_w * scale);
    int scaled_h = (int)std::round(src_h * scale);
    int pad_x    = (net_w - scaled_w) / 2;
    int pad_y    = (net_h - scaled_h) / 2;
    lb_out       = {scale, pad_x, pad_y, src_w, src_h};

    const uint8_t* d_uv = d_nv12 + (size_t)src_w * src_h;

    dim3 block(16, 16);
    dim3 grid((net_w + 15) / 16, (net_h + 15) / 16);
    letterbox_nv12_kernel<<<grid, block, 0, stream>>>(
        d_nv12, d_uv, d_dst,
        src_h, src_w,
        net_h, net_w,
        scale, pad_x, pad_y);
}
