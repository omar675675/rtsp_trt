#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "postprocess.hpp"

// Draw one stream's overlay (green boxes + status label) directly onto its tile
// within the shared canvas.  The canvas already holds the GPU-composited,
// resized frames; this only adds the vector overlay on the CPU.
//
// Boxes are in original source-frame coordinates and are scaled to the cell:
//   canvas_x = col*cell_w + box.x * (cell_w / src_w)
//
// stream_id doubles as the tile index (row = id/cols, col = id%cols).
void draw_tile(cv::Mat& canvas, int stream_id, int cols,
               int cell_w, int cell_h, int src_w, int src_h,
               float fps, const std::vector<Detection>& dets);
