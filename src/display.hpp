#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "postprocess.hpp"

// Draw green bounding boxes and a status bar (stream ID, FPS, face count) in place.
void draw_frame(cv::Mat& frame, int stream_id, float fps,
                const std::vector<Detection>& dets);

// Compose per-stream frames into a tiled canvas.
// Frames that are empty (no frame received yet) are shown as dark grey.
cv::Mat tile_frames(const std::vector<cv::Mat>& frames,
                    int cols, int cell_w, int cell_h);
