#include "pipeline_core.hpp"

#include <iostream>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>

#include <gst/gst.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>

#include "engine.hpp"
#include "stream.hpp"
#include "preprocess.cuh"
#include "postprocess.hpp"
#include "postprocess_gpu.cuh"
#include "crop.cuh"
#include "bytetrack.hpp"

// ── signal handling ───────────────────────────────────────────────────────────

static std::atomic<bool> g_sig_stop{false};
static void sig_handler(int) { g_sig_stop = true; }

// ── config ────────────────────────────────────────────────────────────────────

struct Config {
    std::string engine_path;
    int         engine_max_batch;
    float       conf_thresh, nms_thresh;
    std::vector<std::string> streams;
    int         fps_window;
    int         batch_wait_us;
    bool                tracker_enabled;
    BYTETracker::Config tracker;
};

static Config load_config(const std::string& path) {
    YAML::Node y;
    try { y = YAML::LoadFile(path); }
    catch (const std::exception& e) {
        throw std::runtime_error("Cannot load config: " + std::string(e.what()));
    }
    Config c;
    c.engine_path      = y["model"]["engine"].as<std::string>();
    c.engine_max_batch = y["model"]["engine_max_batch"].as<int>(10);
    c.conf_thresh      = y["detection"]["conf_threshold"].as<float>(0.50f);
    c.nms_thresh       = y["detection"]["nms_threshold"].as<float>(0.45f);
    c.fps_window       = y["fps_window"].as<int>(30);
    c.batch_wait_us    = y["batch_wait_us"].as<int>(3000);
    if (!y["streams"] || !y["streams"].IsSequence())
        throw std::runtime_error("No 'streams' list in config");
    for (const auto& s : y["streams"])
        c.streams.push_back(s.as<std::string>());
    if (c.streams.empty())
        throw std::runtime_error("streams list is empty");

    auto tr = y["tracker"];
    c.tracker_enabled        = tr ? tr["enabled"].as<bool>(true) : true;
    c.tracker.high_thresh    = tr ? tr["high_thresh"].as<float>(0.50f)  : 0.50f;
    c.tracker.low_thresh     = tr ? tr["low_thresh"].as<float>(0.10f)   : 0.10f;
    c.tracker.match_thresh   = tr ? tr["match_thresh"].as<float>(0.80f) : 0.80f;
    c.tracker.max_age        = tr ? tr["max_age"].as<int>(30)           : 30;
    c.tracker.min_hits       = tr ? tr["min_hits"].as<int>(1)           : 1;
    return c;
}

// ── FPS tracker ───────────────────────────────────────────────────────────────

struct FpsTracker {
    std::deque<double> ts;
    int window;
    explicit FpsTracker(int w) : window(w) {}
    void tick() {
        using namespace std::chrono;
        ts.push_back(duration<double>(steady_clock::now().time_since_epoch()).count());
        while ((int)ts.size() > window) ts.pop_front();
    }
    float fps() const {
        if ((int)ts.size() < 2) return 0.f;
        return (float)(ts.size() - 1) / (float)(ts.back() - ts.front());
    }
};

// ── batch collector ───────────────────────────────────────────────────────────

static void collect_batch(
    const std::vector<std::unique_ptr<Stream>>& streams,
    int n, int wait_us, std::vector<Frame>& batch)
{
    batch.clear();
    batch.reserve(n);
    std::vector<bool> got(n, false);
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::microseconds(wait_us);
    do {
        for (int i = 0; i < n; ++i) {
            if (got[i]) continue;
            Frame f;
            if (streams[i]->running() && streams[i]->try_pop(f)) {
                got[i] = true;
                batch.push_back(std::move(f));
            }
        }
        if ((int)batch.size() == n) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
}

static void sync_pre_streams(const std::vector<Frame>& batch, int n,
                              const std::vector<cudaStream_t>& streams)
{
    std::vector<bool> done(n, false);
    for (const auto& f : batch) {
        if (!done[f.stream_id]) {
            cudaStreamSynchronize(streams[f.stream_id]);
            done[f.stream_id] = true;
        }
    }
}

// ── per-stream BGR frame buffers ──────────────────────────────────────────────

struct BgrBuf {
    int      w = 0, h = 0;
    uint8_t* d[2]     = {nullptr, nullptr};
    uint8_t* h_pin[2] = {nullptr, nullptr};

    bool alloc(int width, int height) {
        w = width; h = height;
        for (int s = 0; s < 2; ++s) {
            if (cudaMalloc    (&d[s],     (size_t)w * h * 3) != cudaSuccess) return false;
            if (cudaMallocHost(&h_pin[s], (size_t)w * h * 3) != cudaSuccess) return false;
        }
        return true;
    }
    void free_all() {
        for (int s = 0; s < 2; ++s) {
            if (d[s])     { cudaFree    (d[s]);     d[s]     = nullptr; }
            if (h_pin[s]) { cudaFreeHost(h_pin[s]); h_pin[s] = nullptr; }
        }
        w = h = 0;
    }
};

// ── Pipeline::Impl ────────────────────────────────────────────────────────────

struct Pipeline::Impl {
    Config              cfg;
    StatsCallback       stats_cb;
    DetectionCallback   det_cb;
    std::atomic<bool>   stop_{false};
    std::atomic<bool>   running_{false};
    std::thread         loop_thread_;

    explicit Impl(const std::string& config_path)
        : cfg(load_config(config_path)) {}

    void run_loop(bool handle_signals);
};

// ── Pipeline public API ───────────────────────────────────────────────────────

Pipeline::Pipeline(const std::string& config_path)
    : impl_(std::make_unique<Impl>(config_path))
{
    gst_init(nullptr, nullptr);
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::set_on_stats(StatsCallback cb)     { impl_->stats_cb = std::move(cb); }
void Pipeline::set_on_detection(DetectionCallback cb) { impl_->det_cb = std::move(cb); }

void Pipeline::start() {
    if (impl_->running_.load()) return;
    impl_->stop_ = false;
    impl_->loop_thread_ = std::thread([this]{ impl_->run_loop(false); });
}

void Pipeline::stop(double timeout_sec) {
    impl_->stop_ = true;
    if (impl_->loop_thread_.joinable()) {
        // Use a timed join via a detached waiter thread to honour timeout_sec.
        std::atomic<bool> joined{false};
        std::thread waiter([&]{
            impl_->loop_thread_.join();
            joined = true;
        });
        waiter.detach();
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::duration<double>(timeout_sec);
        while (!joined.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // If still not joined the waiter thread will eventually do so; the
        // loop_thread_ handle is now in an invalid state so reset it.
    }
}

void Pipeline::run() {
    if (impl_->running_.load()) return;
    impl_->stop_ = false;
    impl_->run_loop(true);
}

bool Pipeline::running() const { return impl_->running_.load(); }

// ── run_loop ──────────────────────────────────────────────────────────────────

void Pipeline::Impl::run_loop(bool handle_signals) {
    running_ = true;

    // Optional signal handlers for the blocking (run()) path.
    using SigHandler = void(*)(int);
    SigHandler old_int = SIG_DFL, old_term = SIG_DFL;
    if (handle_signals) {
        g_sig_stop = false;
        old_int  = std::signal(SIGINT,  sig_handler);
        old_term = std::signal(SIGTERM, sig_handler);
    }

    const int n = (int)cfg.streams.size();
    if (n > cfg.engine_max_batch)
        std::cerr << "WARNING: " << n << " streams > engine max batch\n";

    std::cout << "streams: " << n
              << "  batch_wait: " << cfg.batch_wait_us << " µs\n";

    const bool need_crops = (bool)det_cb;

    Engine engine(cfg.engine_path);
    const int net_h       = engine.input_h();
    const int net_w       = engine.input_w();
    const int num_anchors = engine.num_anchors();
    const int out_per_img = engine.num_attrs() * num_anchors;

    // ── GPU buffers ───────────────────────────────────────────────────────────

    float* d_input[2]  = {};
    float* d_output[2] = {};
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_input[i],  (size_t)n * 3 * net_h * net_w * sizeof(float));
        cudaMalloc(&d_output[i], (size_t)n * out_per_img        * sizeof(float));
    }

    Detection* d_dets_gpu       = nullptr;
    int*       d_det_counts_gpu = nullptr;
    cudaMalloc(&d_dets_gpu,       (size_t)n * MAX_DETS_PER_IMG * sizeof(Detection));
    cudaMalloc(&d_det_counts_gpu, (size_t)n * sizeof(int));

    int*       h_det_counts = nullptr;
    Detection* h_dets_all   = nullptr;
    cudaMallocHost(&h_det_counts, (size_t)n * sizeof(int));
    cudaMallocHost(&h_dets_all,   (size_t)n * MAX_DETS_PER_IMG * sizeof(Detection));

    // ── CUDA streams ──────────────────────────────────────────────────────────

    std::vector<cudaStream_t> pre_streams(n);
    for (int i = 0; i < n; ++i)
        cudaStreamCreateWithFlags(&pre_streams[i], cudaStreamNonBlocking);

    cudaStream_t dl_stream   = nullptr;
    cudaStream_t post_stream = nullptr;
    cudaStreamCreateWithFlags(&dl_stream,   cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&post_stream, cudaStreamNonBlocking);

    // ── GStreamer pipelines ───────────────────────────────────────────────────

    std::vector<std::unique_ptr<Stream>> streams;
    streams.reserve(n);
    for (int i = 0; i < n; ++i)
        streams.push_back(std::make_unique<Stream>(i, cfg.streams[i]));
    for (auto& s : streams)
        if (!s->start()) {
            std::cerr << "Failed to start stream " << s->id() << "\n";
            goto cleanup;
        }

    {
        // ── per-stream bookkeeping ────────────────────────────────────────────

        std::vector<FpsTracker>             fps_trackers(n, FpsTracker(cfg.fps_window));
        std::vector<std::vector<Detection>> last_dets(n);
        std::vector<std::vector<TrackedObject>> last_tracked(n);
        std::vector<LetterboxInfo>          lb_infos(n);
        std::vector<BgrBuf>                 bgr_bufs(n);

        std::vector<BYTETracker> trackers;
        if (cfg.tracker_enabled) {
            trackers.reserve(n);
            for (int i = 0; i < n; ++i)
                trackers.emplace_back(cfg.tracker);
        }

        std::cout << "Running — call stop() or Ctrl+C to stop\n";

        int  cur    = 0;
        bool primed = false;
        std::vector<Frame> cur_batch;
        int                cur_sz = 0;

        // ── main loop ─────────────────────────────────────────────────────────

        while (!stop_.load(std::memory_order_relaxed) &&
               !g_sig_stop.load(std::memory_order_relaxed)) {
            for (auto& s : streams) s->poll_bus();

            const int nxt = 1 - cur;

            // ── Phase 1: collect + preprocess NEXT batch ──────────────────────

            std::vector<Frame> nxt_batch;
            collect_batch(streams, n, cfg.batch_wait_us, nxt_batch);
            const bool have_nxt = !nxt_batch.empty();

            if (!have_nxt) {
                bool all_done = true;
                for (const auto& s : streams)
                    if (s->running()) { all_done = false; break; }
                if (all_done) break;
            }

            if (have_nxt) {
                const int nxt_sz = (int)nxt_batch.size();

                for (int b = 0; b < nxt_sz; ++b) {
                    const int sid  = nxt_batch[b].stream_id;
                    float*    slot = d_input[nxt] + (size_t)b * 3 * net_h * net_w;
                    preprocess_frame(
                        nxt_batch[b].d_nv12,
                        nxt_batch[b].src_h, nxt_batch[b].src_w,
                        slot, net_h, net_w,
                        lb_infos[sid], pre_streams[sid]);
                }

                if (need_crops) {
                    for (int b = 0; b < nxt_sz; ++b) {
                        const int sid = nxt_batch[b].stream_id;
                        auto& bb = bgr_bufs[sid];
                        if (bb.w == 0) {
                            if (!bb.alloc(nxt_batch[b].src_w, nxt_batch[b].src_h))
                                std::cerr << "BgrBuf alloc failed stream " << sid << "\n";
                        }
                        if (bb.w > 0) {
                            nv12_to_bgr_frame_gpu(
                                nxt_batch[b].d_nv12, bb.w, bb.h,
                                bb.d[nxt], dl_stream);
                            cudaMemcpyAsync(bb.h_pin[nxt], bb.d[nxt],
                                (size_t)bb.w * bb.h * 3,
                                cudaMemcpyDeviceToHost, dl_stream);
                        }
                    }
                }

                sync_pre_streams(nxt_batch, n, pre_streams);

                if (need_crops)
                    cudaStreamSynchronize(dl_stream);

                for (auto& f : nxt_batch) {
                    streams[f.stream_id]->release_frame(f.d_nv12);
                    f.d_nv12 = nullptr;
                }
            }

            // ── Phase 2: finish cur inference, GPU decode, submit nxt, NMS ────

            if (primed) {
                if (engine.wait() != cudaSuccess) {
                    std::cerr << "TRT sync failed\n"; continue;
                }

                cudaMemsetAsync(d_det_counts_gpu, 0,
                                (size_t)cur_sz * sizeof(int), post_stream);
                for (int b = 0; b < cur_sz; ++b) {
                    const int sid = cur_batch[b].stream_id;
                    decode_detections_gpu(
                        d_output[cur] + (size_t)b * out_per_img,
                        num_anchors, cfg.conf_thresh,
                        lb_infos[sid],
                        cur_batch[b].src_w, cur_batch[b].src_h,
                        d_dets_gpu       + (size_t)b * MAX_DETS_PER_IMG,
                        d_det_counts_gpu + b,
                        post_stream);
                }
                cudaMemcpyAsync(h_det_counts, d_det_counts_gpu,
                                (size_t)cur_sz * sizeof(int),
                                cudaMemcpyDeviceToHost, post_stream);
                cudaMemcpyAsync(h_dets_all, d_dets_gpu,
                                (size_t)cur_sz * MAX_DETS_PER_IMG * sizeof(Detection),
                                cudaMemcpyDeviceToHost, post_stream);

                if (have_nxt)
                    engine.submit(d_input[nxt], d_output[nxt], (int)nxt_batch.size());

                cudaStreamSynchronize(post_stream);

                for (int b = 0; b < cur_sz; ++b) {
                    const int sid = cur_batch[b].stream_id;
                    const int cnt = std::min(h_det_counts[b], MAX_DETS_PER_IMG);
                    const Detection* src = h_dets_all + (size_t)b * MAX_DETS_PER_IMG;
                    std::vector<Detection> dets(src, src + cnt);

                    fps_trackers[sid].tick();
                    last_dets[sid] = nms(std::move(dets), cfg.nms_thresh);

                    if (cfg.tracker_enabled)
                        last_tracked[sid] = trackers[sid].update(last_dets[sid]);

                    if (cur_batch[b].frame_num % (uint64_t)cfg.fps_window == 0 && stats_cb) {
                        StatsEvent ev;
                        ev.stream_id = sid;
                        ev.frame_num = cur_batch[b].frame_num;
                        ev.faces     = cfg.tracker_enabled
                                     ? (int)last_tracked[sid].size()
                                     : (int)last_dets[sid].size();
                        ev.fps       = fps_trackers[sid].fps();
                        stats_cb(ev);
                    }
                }

                // Emit detections with the full frame — Python does the crop.
                // One frame copy per stream per batch; shared across all objects in it.
                if (need_crops) {
                    for (int b = 0; b < cur_sz; ++b) {
                        const int sid = cur_batch[b].stream_id;
                        auto& bb = bgr_bufs[sid];
                        if (bb.w == 0 || !bb.h_pin[cur]) continue;
                        const size_t frame_bytes = (size_t)bb.w * bb.h * 3;
                        auto frame_data = std::make_shared<std::vector<uint8_t>>(
                            bb.h_pin[cur], bb.h_pin[cur] + frame_bytes);

                        // Build unified view: tracked objects (with IDs) or raw detections.
                        const bool use_tracker = cfg.tracker_enabled;
                        const auto& tobjs = last_tracked[sid];
                        const auto& rdets = last_dets[sid];
                        const int   nobj  = use_tracker ? (int)tobjs.size()
                                                        : (int)rdets.size();

                        if (nobj == 0) {
                            // No objects — fire once with zeros so Python still gets the frame.
                            DetectionEvent ev;
                            ev.stream_id = sid;
                            ev.frame_num = cur_batch[b].frame_num;
                            ev.x = ev.y = ev.w = ev.h = ev.conf = 0.f;
                            ev.track_id  = 0;
                            ev.frame_w = bb.w; ev.frame_h = bb.h;
                            ev.frame_bgr = frame_data;
                            det_cb(ev);
                        } else if (use_tracker) {
                            for (const auto& obj : tobjs) {
                                DetectionEvent ev;
                                ev.stream_id = sid;
                                ev.frame_num = cur_batch[b].frame_num;
                                ev.x = obj.x; ev.y = obj.y;
                                ev.w = obj.w; ev.h = obj.h;
                                ev.conf      = obj.conf;
                                ev.track_id  = obj.track_id;
                                ev.frame_w = bb.w; ev.frame_h = bb.h;
                                ev.frame_bgr = frame_data;
                                det_cb(ev);
                            }
                        } else {
                            for (const auto& det : rdets) {
                                DetectionEvent ev;
                                ev.stream_id = sid;
                                ev.frame_num = cur_batch[b].frame_num;
                                ev.x = det.left; ev.y = det.top;
                                ev.w = det.width; ev.h = det.height;
                                ev.conf      = det.conf;
                                ev.track_id  = 0;
                                ev.frame_w = bb.w; ev.frame_h = bb.h;
                                ev.frame_bgr = frame_data;
                                det_cb(ev);
                            }
                        }
                    }
                }

                if (have_nxt) {
                    cur       = nxt;
                    cur_batch = std::move(nxt_batch);
                    cur_sz    = (int)cur_batch.size();
                } else {
                    primed = false;
                }

            } else if (have_nxt) {
                engine.submit(d_input[nxt], d_output[nxt], (int)nxt_batch.size());
                cur       = nxt;
                cur_batch = std::move(nxt_batch);
                cur_sz    = (int)cur_batch.size();
                primed    = true;

            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        // ── drain final batch ─────────────────────────────────────────────────
        if (primed) {
            engine.wait();
            cudaMemsetAsync(d_det_counts_gpu, 0, (size_t)cur_sz * sizeof(int), post_stream);
            for (int b = 0; b < cur_sz; ++b) {
                int sid = cur_batch[b].stream_id;
                decode_detections_gpu(
                    d_output[cur] + (size_t)b * out_per_img,
                    num_anchors, cfg.conf_thresh, lb_infos[sid],
                    cur_batch[b].src_w, cur_batch[b].src_h,
                    d_dets_gpu + (size_t)b * MAX_DETS_PER_IMG,
                    d_det_counts_gpu + b, post_stream);
            }
            cudaMemcpyAsync(h_det_counts, d_det_counts_gpu,
                            cur_sz * sizeof(int), cudaMemcpyDeviceToHost, post_stream);
            cudaMemcpyAsync(h_dets_all, d_dets_gpu,
                            (size_t)cur_sz * MAX_DETS_PER_IMG * sizeof(Detection),
                            cudaMemcpyDeviceToHost, post_stream);
            cudaStreamSynchronize(post_stream);
        }

        std::cout << "Stopping...\n";
        for (auto& s : streams) s->stop();
        for (auto& bb : bgr_bufs) bb.free_all();
    }

cleanup:
    for (int i = 0; i < n; ++i) cudaStreamDestroy(pre_streams[i]);
    cudaStreamDestroy(dl_stream);
    cudaStreamDestroy(post_stream);
    for (int i = 0; i < 2; ++i) { cudaFree(d_input[i]); cudaFree(d_output[i]); }
    cudaFree(d_dets_gpu); cudaFree(d_det_counts_gpu);
    cudaFreeHost(h_det_counts); cudaFreeHost(h_dets_all);

    if (handle_signals) {
        std::signal(SIGINT,  old_int);
        std::signal(SIGTERM, old_term);
    }

    running_ = false;
}
