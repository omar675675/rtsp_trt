#include "postprocess.hpp"
#include <algorithm>
#include <cmath>

std::vector<Detection> decode_detections(
    const float*        output,
    int                 num_attrs,
    int                 num_anchors,
    float               conf_thresh,
    const LetterboxInfo& lb,
    int                 frame_w,
    int                 frame_h)
{
    (void)num_attrs;    // always ≥ 5; silences -Wunused-parameter
    std::vector<Detection> dets;
    dets.reserve(64);

    // Tensor layout: output[attr * num_anchors + a]
    for (int a = 0; a < num_anchors; ++a) {
        float score = output[4 * num_anchors + a];
        if (score < conf_thresh) continue;

        float cx = output[0 * num_anchors + a];
        float cy = output[1 * num_anchors + a];
        float w  = output[2 * num_anchors + a];
        float h  = output[3 * num_anchors + a];

        // Inverse letterbox: network coords → original frame coords
        float left   = (cx - w * 0.5f - lb.pad_x) / lb.scale;
        float top    = (cy - h * 0.5f - lb.pad_y) / lb.scale;
        float right  = (cx + w * 0.5f - lb.pad_x) / lb.scale;
        float bottom = (cy + h * 0.5f - lb.pad_y) / lb.scale;

        left   = std::max(0.f, std::min(left,   (float)(frame_w - 1)));
        top    = std::max(0.f, std::min(top,    (float)(frame_h - 1)));
        right  = std::max(0.f, std::min(right,  (float)(frame_w - 1)));
        bottom = std::max(0.f, std::min(bottom, (float)(frame_h - 1)));

        float bw = right - left, bh = bottom - top;
        if (bw <= 0.f || bh <= 0.f) continue;

        dets.push_back({left, top, bw, bh, score});
    }
    return dets;
}

static float iou(const Detection& a, const Detection& b) {
    float ax2 = a.left + a.width,  ay2 = a.top + a.height;
    float bx2 = b.left + b.width,  by2 = b.top + b.height;
    float ix1 = std::max(a.left, b.left), iy1 = std::max(a.top, b.top);
    float ix2 = std::min(ax2, bx2),       iy2 = std::min(ay2, by2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float ua = a.width * a.height + b.width * b.height - inter;
    return ua > 0.f ? inter / ua : 0.f;
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b){ return a.conf > b.conf; });

    std::vector<bool> dead(dets.size(), false);
    std::vector<Detection> out;
    out.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!dead[j] && iou(dets[i], dets[j]) > iou_thresh)
                dead[j] = true;
    }
    return out;
}
