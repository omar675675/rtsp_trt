#pragma once
#include "postprocess.hpp"
#include <memory>
#include <vector>

// Output of BYTETracker::update() — one entry per live tracked object.
struct TrackedObject {
    float x, y, w, h;  // Kalman-smoothed position in original frame coords (left/top/w/h)
    float conf;         // detection confidence at last match
    int   track_id;     // persistent ID (> 0)
};

// ByteTrack: two-stage IoU association with Kalman prediction.
// One instance per stream; call update() once per frame with NMS detections.
class BYTETracker {
public:
    struct Config {
        float high_thresh  = 0.50f;  // min conf to enter high-quality detection pool
        float low_thresh   = 0.10f;  // min conf for low-quality pool (stage-2 matching)
        float match_thresh = 0.80f;  // IoU threshold for high-quality matching (stages 1 & 3)
        float refind_dist  = 2.5f;   // max centre distance (in box-radii) to re-find a lost track
        int   max_age      = 30;     // frames a lost track survives without a match
        int   min_hits     = 1;      // consecutive hits before a track appears in output
    };

    explicit BYTETracker(const Config& cfg);
    ~BYTETracker();
    BYTETracker(BYTETracker&&) noexcept;
    BYTETracker& operator=(BYTETracker&&) noexcept;

    // Feed per-frame NMS detections; returns all live tracked objects.
    std::vector<TrackedObject> update(const std::vector<Detection>& dets);

    // Clear all tracks (call on stream reconnect / seek).
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
