#pragma once
#include <cuda_runtime.h>
#include "postprocess.hpp"   // Detection, LetterboxInfo (via postprocess.hpp→preprocess.cuh)

// Maximum detections written per image before the atomic counter saturates.
// Face detection at 0.5 threshold rarely exceeds 100 even in dense crowds.
static constexpr int MAX_DETS_PER_IMG = 200;

// GPU confidence filter + inverse-letterbox decode.
//
// Reads the raw TRT output tensor for ONE image (d_output points to the
// [num_attrs × num_anchors] slice for that image) and writes passing detections
// into a compacted device array using atomicAdd.
//
// d_count MUST be zeroed (cudaMemsetAsync) before each call.
// Runs asynchronously on stream — caller must sync before reading d_out / d_count.
void decode_detections_gpu(
    const float*  d_output,    // device ptr [num_attrs × num_anchors] — one image
    int           num_anchors,
    float         conf_thresh,
    LetterboxInfo lb,           // passed by value (small, lives in registers)
    int           frame_w,
    int           frame_h,
    Detection*    d_out,        // device ptr [MAX_DETS_PER_IMG] — pre-allocated
    int*          d_count,      // device ptr [1] atomic counter — must be pre-zeroed
    cudaStream_t  stream);
