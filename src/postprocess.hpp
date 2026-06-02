#pragma once
#include <vector>
#include "preprocess.cuh"   // LetterboxInfo

struct Detection {
    float left, top, width, height;
    float conf;
};

// Decode one image's slice of the output0 tensor [num_attrs × num_anchors]
// (layout: attr-major, i.e. value(attr,a) = buf[attr*num_anchors + a])
// and map boxes from network-input coords back to original frame coords.
std::vector<Detection> decode_detections(
    const float*        output,      // [num_attrs * num_anchors] for this image
    int                 num_attrs,
    int                 num_anchors,
    float               conf_thresh,
    const LetterboxInfo& lb,
    int                 frame_w,
    int                 frame_h);

// Greedy IoU NMS (in-place sort by confidence).
std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh);
