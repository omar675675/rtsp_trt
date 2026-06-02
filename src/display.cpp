#include "display.hpp"
#include <sstream>
#include <iomanip>

void draw_tile(cv::Mat& canvas, int stream_id, int cols,
               int cell_w, int cell_h, int src_w, int src_h,
               float fps, const std::vector<Detection>& dets)
{
    const int   col = stream_id % cols;
    const int   row = stream_id / cols;
    const int   ox  = col * cell_w;
    const int   oy  = row * cell_h;
    const float sx  = (float)cell_w / (float)src_w;
    const float sy  = (float)cell_h / (float)src_h;

    for (const auto& d : dets) {
        cv::rectangle(canvas,
            cv::Rect((int)(ox + d.left  * sx), (int)(oy + d.top    * sy),
                     (int)(d.width * sx),      (int)(d.height * sy)),
            cv::Scalar(0, 255, 0), 2);
    }

    std::ostringstream oss;
    oss << "Stream " << stream_id
        << "  FPS " << std::fixed << std::setprecision(1) << fps
        << "  Faces " << dets.size();
    const std::string label = oss.str();

    cv::putText(canvas, label, {ox + 10, oy + 26}, cv::FONT_HERSHEY_SIMPLEX,
                0.65, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(canvas, label, {ox + 10, oy + 26}, cv::FONT_HERSHEY_SIMPLEX,
                0.65, {255, 255, 255}, 1, cv::LINE_AA);
}
