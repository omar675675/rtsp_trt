#include "postprocess_gpu.cuh"

// One thread per anchor.  Threads that pass the confidence threshold do an
// atomicAdd to grab an output slot and write one Detection struct directly into
// device memory — no D2H transfer of the 672 KB raw output tensor needed.

__global__ void decode_kernel(
    const float* __restrict__ d_output,
    int   num_anchors,
    float conf_thresh,
    float scale, int pad_x, int pad_y,
    int   frame_w, int frame_h,
    Detection* d_out,
    int*       d_count)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= num_anchors) return;

    float score = d_output[4 * num_anchors + a];
    if (score < conf_thresh) return;

    float cx = d_output[0 * num_anchors + a];
    float cy = d_output[1 * num_anchors + a];
    float bw = d_output[2 * num_anchors + a];
    float bh = d_output[3 * num_anchors + a];

    float fw = (float)(frame_w - 1);
    float fh = (float)(frame_h - 1);
    float l = fmaxf(0.f, fminf((cx - bw * 0.5f - pad_x) / scale, fw));
    float t = fmaxf(0.f, fminf((cy - bh * 0.5f - pad_y) / scale, fh));
    float r = fmaxf(0.f, fminf((cx + bw * 0.5f - pad_x) / scale, fw));
    float b = fmaxf(0.f, fminf((cy + bh * 0.5f - pad_y) / scale, fh));

    float dw = r - l, dh = b - t;
    if (dw <= 0.f || dh <= 0.f) return;

    int idx = atomicAdd(d_count, 1);
    if (idx < MAX_DETS_PER_IMG)
        d_out[idx] = {l, t, dw, dh, score};
}

void decode_detections_gpu(
    const float* d_output, int num_anchors, float conf_thresh,
    LetterboxInfo lb, int frame_w, int frame_h,
    Detection* d_out, int* d_count, cudaStream_t stream)
{
    int blocks = (num_anchors + 255) / 256;
    decode_kernel<<<blocks, 256, 0, stream>>>(
        d_output, num_anchors, conf_thresh,
        lb.scale, lb.pad_x, lb.pad_y,
        frame_w, frame_h,
        d_out, d_count);
}
