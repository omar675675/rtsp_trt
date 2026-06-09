#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class Pipeline {
public:
    struct StatsEvent {
        int      stream_id;
        uint64_t frame_num;
        int      faces;
        float    fps;
    };

    struct DetectionEvent {
        int      stream_id;
        uint64_t frame_num;
        float    x, y, w, h, conf;
        int      track_id;   // persistent object ID from ByteTracker (0 if tracking disabled)
        int      frame_w, frame_h;
        // Full BGR frame shared across all detections in the same frame.
        // One copy per frame regardless of how many objects are detected.
        std::shared_ptr<const std::vector<uint8_t>> frame_bgr;
    };

    using StatsCallback     = std::function<void(const StatsEvent&)>;
    using DetectionCallback = std::function<void(const DetectionEvent&)>;

    explicit Pipeline(const std::string& config_path);
    ~Pipeline();

    void set_on_stats(StatsCallback cb);
    void set_on_detection(DetectionCallback cb);

    // Non-blocking: launch the pipeline loop in a background thread.
    void start();
    // Signal the loop to stop and wait for it to finish.
    void stop(double timeout_sec = 5.0);
    // Blocking: run in the calling thread; installs/restores SIGINT+SIGTERM.
    void run();

    bool running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
