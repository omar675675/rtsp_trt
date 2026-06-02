#include "display.hpp"
#include <sstream>
#include <iomanip>

void draw_frame(cv::Mat& frame, int stream_id, float fps,
                const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        cv::rectangle(frame,
            cv::Rect((int)d.left, (int)d.top, (int)d.width, (int)d.height),
            cv::Scalar(0, 255, 0), 2);
    }

    std::ostringstream oss;
    oss << "Stream " << stream_id
        << "  FPS " << std::fixed << std::setprecision(1) << fps
        << "  Faces " << dets.size();
    const std::string label = oss.str();

    // Shadow + white text
    cv::putText(frame, label, {10, 26}, cv::FONT_HERSHEY_SIMPLEX,
                0.65, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(frame, label, {10, 26}, cv::FONT_HERSHEY_SIMPLEX,
                0.65, {255, 255, 255}, 1, cv::LINE_AA);
}

cv::Mat tile_frames(const std::vector<cv::Mat>& frames,
                    int cols, int cell_w, int cell_h)
{
    int n    = (int)frames.size();
    int rows = (n + cols - 1) / cols;
    cv::Mat canvas(rows * cell_h, cols * cell_w, CV_8UC3,
                   cv::Scalar(30, 30, 30));

    for (int i = 0; i < n; ++i) {
        int r = i / cols, c = i % cols;
        cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);
        if (frames[i].empty()) continue;
        cv::Mat dst = canvas(roi);
        cv::resize(frames[i], dst, {cell_w, cell_h});
    }
    return canvas;
}
