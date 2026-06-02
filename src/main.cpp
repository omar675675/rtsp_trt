/*
 * rtsp_trt — multi-stream face detection pipeline
 *
 * Architecture matches nvinfer/nvstreammux internally:
 *
 *   nvh264dec → CUDAMemory NV12
 *       ↓  [non-blocking D2D copy in callback, copy_stream per stream]
 *   Pool buffer (our primary CUDA context, packed NV12)
 *       ↓  [letterbox + NV12→RGB kernel, per-stream non-blocking pre_stream]
 *   d_input[nxt]  ←── preprocess (concurrent with TRT on d_input[cur])
 *       ↓  [engine.submit → enqueueV3 async]
 *   TRT inference
 *       ↓  [engine.wait → cudaStreamSynchronize]
 *   h_output ← cudaMemcpy  (before engine.submit(nxt) — null stream issue avoided)
 *       ↓
 *   CPU: decode_detections + NMS  (concurrent with new TRT inference)
 */

#include <iostream>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <atomic>
#include <csignal>
#include <iomanip>

#include <gst/gst.h>
#include <cuda_runtime.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "engine.hpp"
#include "stream.hpp"
#include "preprocess.cuh"
#include "postprocess.hpp"
#include "display.hpp"

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// ── config ────────────────────────────────────────────────────────────────────

struct Config {
    bool        display;
    std::string engine_path;
    int         input_w, input_h, engine_max_batch;
    float       conf_thresh, nms_thresh;
    std::vector<std::string> streams;
    int         tiler_cols, cell_w, cell_h;
    int         fps_window;
    int         batch_wait_us;  // spin-wait budget when collecting a batch (µs)
};

static Config load_config(const std::string& path) {
    YAML::Node y;
    try { y = YAML::LoadFile(path); }
    catch (const std::exception& e) {
        throw std::runtime_error("Cannot load config: " + std::string(e.what()));
    }
    Config c;
    c.display          = y["display"].as<bool>(false);
    c.engine_path      = y["model"]["engine"].as<std::string>();
    c.input_w          = y["model"]["input_width"].as<int>(640);
    c.input_h          = y["model"]["input_height"].as<int>(640);
    c.engine_max_batch = y["model"]["engine_max_batch"].as<int>(10);
    c.conf_thresh      = y["detection"]["conf_threshold"].as<float>(0.50f);
    c.nms_thresh       = y["detection"]["nms_threshold"].as<float>(0.45f);
    c.tiler_cols       = y["tiler"]["columns"].as<int>(2);
    c.cell_w           = y["tiler"]["cell_width"].as<int>(640);
    c.cell_h           = y["tiler"]["cell_height"].as<int>(360);
    c.fps_window       = y["fps_window"].as<int>(30);
    // Equivalent to nvstreammux batched_push_timeout but far shorter: we just
    // spin briefly to let all streams contribute to the batch before inferring.
    // Default 3 ms.  Set to 0 to disable (greedy collect, like before).
    c.batch_wait_us    = y["batch_wait_us"].as<int>(3000);
    if (!y["streams"] || !y["streams"].IsSequence())
        throw std::runtime_error("No 'streams' list in config");
    for (const auto& s : y["streams"])
        c.streams.push_back(s.as<std::string>());
    if (c.streams.empty())
        throw std::runtime_error("streams list is empty");
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

// ── batch collection ──────────────────────────────────────────────────────────
// Mirrors nvstreammux: spin briefly to give all streams a chance to contribute
// a frame before we commit to the batch.  Ensures TRT always sees a full batch,
// maximising GPU utilisation.  Once the deadline passes we take whatever we have.

static void collect_batch(
    const std::vector<std::unique_ptr<Stream>>& streams,
    int n, int wait_us,
    std::vector<Frame>& batch)
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

// ── helpers ───────────────────────────────────────────────────────────────────

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

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    const char* env_cfg = std::getenv("RTSP_TRT_CONFIG");
    std::string cfg_path = argc > 1 ? argv[1]
                         : (env_cfg ? env_cfg : "config.yaml");
    Config cfg;
    try { cfg = load_config(cfg_path); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }

    const int n = (int)cfg.streams.size();
    if (n > cfg.engine_max_batch)
        std::cerr << "WARNING: " << n << " streams > engine max batch "
                  << cfg.engine_max_batch << "\n";

    std::cout << cfg_path << "  streams: " << n
              << "  display: " << cfg.display
              << "  batch_wait: " << cfg.batch_wait_us << " µs\n";

    gst_init(nullptr, nullptr);

    Engine engine(cfg.engine_path);
    const int net_h       = engine.input_h();
    const int net_w       = engine.input_w();
    const int num_attrs   = engine.num_attrs();
    const int num_anchors = engine.num_anchors();
    const int out_per_img = num_attrs * num_anchors;

    // Double-buffered GPU I/O — while TRT runs on buf[cur], we preprocess into buf[nxt].
    float* d_input[2]  = {nullptr, nullptr};
    float* d_output[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_input[i],  (size_t)n * 3 * net_h * net_w * sizeof(float));
        cudaMalloc(&d_output[i], (size_t)n * out_per_img        * sizeof(float));
    }
    std::vector<float> h_output[2];
    h_output[0].resize((size_t)n * out_per_img);
    h_output[1].resize((size_t)n * out_per_img);

    // Per-stream preprocessing streams — non-blocking so they never serialise
    // with TRT's engine stream or the null stream.
    std::vector<cudaStream_t> pre_streams(n);
    for (int i = 0; i < n; ++i)
        cudaStreamCreateWithFlags(&pre_streams[i], cudaStreamNonBlocking);

    // Non-blocking stream for display downloads.
    cudaStream_t dl_stream = nullptr;
    if (cfg.display)
        cudaStreamCreateWithFlags(&dl_stream, cudaStreamNonBlocking);

    // GStreamer decode pipelines.
    std::vector<std::unique_ptr<Stream>> streams;
    streams.reserve(n);
    for (int i = 0; i < n; ++i)
        streams.push_back(std::make_unique<Stream>(i, cfg.streams[i]));
    for (auto& s : streams)
        if (!s->start()) {
            std::cerr << "Failed to start stream " << s->id() << "\n"; return 1;
        }

    std::vector<FpsTracker>             fps_trackers(n, FpsTracker(cfg.fps_window));
    std::vector<std::vector<Detection>> last_dets(n);
    std::vector<LetterboxInfo>          lb_infos(n);

    // Display buffers — only allocated when display=true.
    std::vector<cv::Mat> nv12_mats(cfg.display ? n : 0);
    std::vector<cv::Mat> bgr_mats(cfg.display ? n : 0);

    std::cout << "Running — Ctrl+C to stop\n";

    int  cur    = 0;
    bool primed = false;
    std::vector<Frame> cur_batch;
    int                cur_sz = 0;

    while (!g_stop) {
        for (auto& s : streams) s->poll_bus();

        const int nxt = 1 - cur;

        // ── Phase 1: collect + preprocess NEXT batch ──────────────────────────
        // Runs concurrently with TRT inference for cur_batch (if primed).
        // Pool buffers are in our primary CUDA context — same as TRT — so the
        // preprocess kernel has no cross-context overhead.

        std::vector<Frame> nxt_batch;
        collect_batch(streams, n, cfg.batch_wait_us, nxt_batch);
        const bool have_nxt = !nxt_batch.empty();

        if (have_nxt) {
            const int nxt_sz = (int)nxt_batch.size();

            for (int b = 0; b < nxt_sz; ++b) {
                const int sid  = nxt_batch[b].stream_id;
                float*    slot = d_input[nxt] + (size_t)b * 3 * net_h * net_w;
                preprocess_frame(
                    nxt_batch[b].d_nv12,
                    nxt_batch[b].src_h, nxt_batch[b].src_w,
                    slot, net_h, net_w,
                    lb_infos[sid],
                    pre_streams[sid]);
            }

            sync_pre_streams(nxt_batch, n, pre_streams);

            // Display: download packed NV12 from pool buffer on non-blocking dl_stream.
            // Concurrent with TRT inference on cur batch.
            if (cfg.display) {
                for (int b = 0; b < nxt_sz; ++b) {
                    const int sid = nxt_batch[b].stream_id;
                    const int W   = nxt_batch[b].src_w;
                    const int H   = nxt_batch[b].src_h;
                    cv::Mat& nvm  = nv12_mats[sid];
                    if (nvm.empty()) nvm.create(H * 3 / 2, W, CV_8UC1);
                    // Pool buffer is packed (stride == W) — single contiguous copy.
                    cudaMemcpyAsync(nvm.data, nxt_batch[b].d_nv12,
                                    (size_t)W * H * 3 / 2,
                                    cudaMemcpyDeviceToHost, dl_stream);
                }
                cudaStreamSynchronize(dl_stream);
            }

            // Return pool buffers — preprocess kernels synced, display done.
            for (auto& f : nxt_batch) {
                streams[f.stream_id]->release_frame(f.d_nv12);
                f.d_nv12 = nullptr;
            }
        }

        // ── Phase 2: finish cur inference, download, submit nxt, then NMS ────

        if (primed) {
            if (engine.wait() != cudaSuccess) {
                std::cerr << "TRT sync failed\n";
                continue;
            }

            // Download BEFORE submitting the next batch.
            // engine.stream_ is idle after wait() so the null stream doesn't
            // block.  If we submitted first, the null stream cudaMemcpy would
            // stall until the new inference finishes — negating the pipeline.
            cudaMemcpy(h_output[cur].data(), d_output[cur],
                       (size_t)cur_sz * out_per_img * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // Restart TRT on the next batch immediately.
            if (have_nxt)
                engine.submit(d_input[nxt], d_output[nxt], (int)nxt_batch.size());

            // NMS + stats — CPU work, runs concurrently with the new TRT above.
            for (int b = 0; b < cur_sz; ++b) {
                const int sid     = cur_batch[b].stream_id;
                const auto& lb    = lb_infos[sid];
                const float* base = h_output[cur].data() + (size_t)b * out_per_img;

                auto dets = decode_detections(base, num_attrs, num_anchors,
                                              cfg.conf_thresh, lb,
                                              cur_batch[b].src_w, cur_batch[b].src_h);
                dets = nms(std::move(dets), cfg.nms_thresh);

                fps_trackers[sid].tick();
                last_dets[sid] = std::move(dets);

                if (cur_batch[b].frame_num % (uint64_t)cfg.fps_window == 0) {
                    std::cout << "[src=" << sid
                              << " frame=" << cur_batch[b].frame_num
                              << "] faces=" << last_dets[sid].size()
                              << " fps=" << std::fixed << std::setprecision(1)
                              << fps_trackers[sid].fps() << "\n";
                }
            }

            // Display render — also concurrent with the new TRT.
            if (cfg.display) {
                for (int b = 0; b < cur_sz; ++b) {
                    int sid = cur_batch[b].stream_id;
                    if (!nv12_mats[sid].empty())
                        cv::cvtColor(nv12_mats[sid], bgr_mats[sid],
                                     cv::COLOR_YUV2BGR_NV12);
                    draw_frame(bgr_mats[sid], sid,
                               fps_trackers[sid].fps(), last_dets[sid]);
                }
                cv::Mat canvas = tile_frames(bgr_mats,
                                             cfg.tiler_cols, cfg.cell_w, cfg.cell_h);
                cv::imshow("rtsp_trt", canvas);
                if (cv::waitKey(1) == 27) break;
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

    // Drain the final in-flight batch.
    if (primed) {
        engine.wait();
        cudaMemcpy(h_output[cur].data(), d_output[cur],
                   (size_t)cur_sz * out_per_img * sizeof(float),
                   cudaMemcpyDeviceToHost);
        for (int b = 0; b < cur_sz; ++b) {
            int sid  = cur_batch[b].stream_id;
            auto dets = decode_detections(
                h_output[cur].data() + (size_t)b * out_per_img,
                num_attrs, num_anchors, cfg.conf_thresh,
                lb_infos[sid], cur_batch[b].src_w, cur_batch[b].src_h);
            last_dets[sid] = nms(std::move(dets), cfg.nms_thresh);
        }
    }

    std::cout << "Stopping...\n";
    for (auto& s : streams) s->stop();
    cv::destroyAllWindows();
    for (int i = 0; i < n; ++i) cudaStreamDestroy(pre_streams[i]);
    if (dl_stream) cudaStreamDestroy(dl_stream);
    for (int i = 0; i < 2; ++i) { cudaFree(d_input[i]); cudaFree(d_output[i]); }
    return 0;
}
