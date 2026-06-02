/*
 * rtsp_trt — multi-stream face detection pipeline
 *
 * Hot-path summary (all GPU, no CPU pixel work, no sequential stalls):
 *
 *  ┌─ Phase 1 (concurrent with TRT inference on cur batch) ────────────────┐
 *  │  collect_batch (spin ≤ batch_wait_us)                                 │
 *  │  preprocess kernels    → d_input[nxt]   (pre_streams, non-blocking)   │
 *  │  nv12_to_bgr kernels   → d_bgr_gpu[sid] (dl_stream,  non-blocking)   │
 *  │  BGR download          → bgr_mats[sid]  (dl_stream,  non-blocking)   │
 *  │  sync pre_streams + dl_stream                                         │
 *  │  release pool buffers                                                 │
 *  └───────────────────────────────────────────────────────────────────────┘
 *
 *  ┌─ Phase 2 ─────────────────────────────────────────────────────────────┐
 *  │  engine.wait()                     TRT done, d_output[cur] ready      │
 *  │  decode_detections_gpu  (post_stream, non-blocking)                   │
 *  │    cudaMemcpyAsync counts + dets   (post_stream, pinned host)         │
 *  │  engine.submit(nxt)                TRT restarts immediately           │
 *  │  cudaStreamSynchronize(post_stream) ← ~0.1 ms, TRT still running     │
 *  │  std::async NMS × batch_sz         (CPU threads, concurrent with TRT) │
 *  │  join NMS futures + print stats                                       │
 *  │  draw + tile + imshow + cv::pollKey (CPU, concurrent with TRT)        │
 *  └───────────────────────────────────────────────────────────────────────┘
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
#include "postprocess_gpu.cuh"
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
    int         batch_wait_us;
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
        std::cerr << "WARNING: " << n << " streams > engine max batch\n";

    std::cout << cfg_path << "  streams: " << n
              << "  display: " << cfg.display
              << "  batch_wait: " << cfg.batch_wait_us << " µs\n";

    gst_init(nullptr, nullptr);

    Engine engine(cfg.engine_path);
    const int net_h       = engine.input_h();
    const int net_w       = engine.input_w();
    const int num_anchors = engine.num_anchors();
    const int out_per_img = engine.num_attrs() * num_anchors;

    // ── GPU buffers ───────────────────────────────────────────────────────────

    // Double-buffered TRT I/O
    float* d_input[2]  = {};
    float* d_output[2] = {};
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_input[i],  (size_t)n * 3 * net_h * net_w * sizeof(float));
        cudaMalloc(&d_output[i], (size_t)n * out_per_img        * sizeof(float));
    }

    // GPU decode output (single — decoded after engine.wait, before submit)
    Detection* d_dets_gpu  = nullptr;
    int*       d_det_counts_gpu = nullptr;
    cudaMalloc(&d_dets_gpu,       (size_t)n * MAX_DETS_PER_IMG * sizeof(Detection));
    cudaMalloc(&d_det_counts_gpu, (size_t)n * sizeof(int));

    // Pinned host memory for fast async D2H of decode results.
    // Pinned = DMA-able by CUDA without staging copies.
    int*       h_det_counts = nullptr;
    Detection* h_dets_all   = nullptr;
    cudaMallocHost(&h_det_counts, (size_t)n * sizeof(int));
    cudaMallocHost(&h_dets_all,   (size_t)n * MAX_DETS_PER_IMG * sizeof(Detection));

    // ── display compositor buffers ────────────────────────────────────────────
    // The GPU composes all tiles into one canvas; we download only that canvas.
    const int cols      = cfg.display ? std::max(1, std::min(cfg.tiler_cols, n)) : 0;
    const int rows      = cfg.display ? (n + cols - 1) / cols : 0;
    const int canvas_w  = cfg.display ? cols * cfg.cell_w : 0;
    const int canvas_h  = cfg.display ? rows * cfg.cell_h : 0;

    uint8_t* d_canvas = nullptr;            // GPU canvas (BGR HWC)
    cv::Mat  h_canvas[2];                   // double-buffered host canvas
    if (cfg.display) {
        cudaMalloc(&d_canvas, (size_t)canvas_w * canvas_h * 3);
        // Clear once to dark grey; cells for live streams are overwritten every
        // frame, empty cells (partial last row) stay grey.
        cudaMemset(d_canvas, 30, (size_t)canvas_w * canvas_h * 3);
        h_canvas[0].create(canvas_h, canvas_w, CV_8UC3);
        h_canvas[1].create(canvas_h, canvas_w, CV_8UC3);
    }

    // ── CUDA streams ──────────────────────────────────────────────────────────
    // All non-blocking so they never serialise with each other or the null stream.

    std::vector<cudaStream_t> pre_streams(n);
    for (int i = 0; i < n; ++i)
        cudaStreamCreateWithFlags(&pre_streams[i], cudaStreamNonBlocking);

    cudaStream_t dl_stream   = nullptr;   // display D2H
    cudaStream_t post_stream = nullptr;   // GPU decode + tiny D2H
    cudaStreamCreateWithFlags(&dl_stream,   cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&post_stream, cudaStreamNonBlocking);

    // ── GStreamer pipelines ───────────────────────────────────────────────────

    std::vector<std::unique_ptr<Stream>> streams;
    streams.reserve(n);
    for (int i = 0; i < n; ++i)
        streams.push_back(std::make_unique<Stream>(i, cfg.streams[i]));
    for (auto& s : streams)
        if (!s->start()) {
            std::cerr << "Failed to start stream " << s->id() << "\n"; return 1;
        }

    // ── per-stream bookkeeping ────────────────────────────────────────────────

    std::vector<FpsTracker>             fps_trackers(n, FpsTracker(cfg.fps_window));
    std::vector<std::vector<Detection>> last_dets(n);
    std::vector<LetterboxInfo>          lb_infos(n);

    std::cout << "Running — Ctrl+C to stop\n";

    // ── pipeline state ────────────────────────────────────────────────────────

    int  cur    = 0;
    bool primed = false;
    std::vector<Frame> cur_batch;
    int                cur_sz = 0;

    // ── main loop ─────────────────────────────────────────────────────────────
    while (!g_stop) {
        for (auto& s : streams) s->poll_bus();

        const int nxt = 1 - cur;

        // ── Phase 1: collect + preprocess NEXT batch ──────────────────────────
        // Runs fully concurrent with TRT inference (if primed).

        std::vector<Frame> nxt_batch;
        collect_batch(streams, n, cfg.batch_wait_us, nxt_batch);
        const bool have_nxt = !nxt_batch.empty();

        if (have_nxt) {
            const int nxt_sz = (int)nxt_batch.size();

            // Preprocess kernels (non-blocking pre_streams)
            for (int b = 0; b < nxt_sz; ++b) {
                const int sid  = nxt_batch[b].stream_id;
                float*    slot = d_input[nxt] + (size_t)b * 3 * net_h * net_w;
                preprocess_frame(
                    nxt_batch[b].d_nv12,
                    nxt_batch[b].src_h, nxt_batch[b].src_w,
                    slot, net_h, net_w,
                    lb_infos[sid], pre_streams[sid]);
            }

            // GPU display compositor (dl_stream, non-blocking): each stream's
            // frame is resized + colour-converted + placed into its tile of the
            // shared GPU canvas.  Only the single composed canvas is downloaded —
            // not N full-resolution frames.  Written into h_canvas[nxt] so it
            // pairs with this batch's detections one iteration later (Phase 2
            // displays h_canvas[cur], keeping image and boxes in sync).
            if (cfg.display) {
                for (int b = 0; b < nxt_sz; ++b) {
                    const int sid = nxt_batch[b].stream_id;
                    const int col = sid % cols, row = sid / cols;
                    nv12_to_bgr_tile_gpu(
                        nxt_batch[b].d_nv12, nxt_batch[b].src_w, nxt_batch[b].src_h,
                        d_canvas, canvas_w,
                        col * cfg.cell_w, row * cfg.cell_h,
                        cfg.cell_w, cfg.cell_h,
                        dl_stream);
                }
                cudaMemcpyAsync(h_canvas[nxt].data, d_canvas,
                                (size_t)canvas_w * canvas_h * 3,
                                cudaMemcpyDeviceToHost, dl_stream);
            }

            // Sync preprocessing — kernels done, safe to release pool buffers
            sync_pre_streams(nxt_batch, n, pre_streams);

            // Sync display compositor + download
            if (cfg.display)
                cudaStreamSynchronize(dl_stream);

            // Return pool NV12 buffers to streams
            for (auto& f : nxt_batch) {
                streams[f.stream_id]->release_frame(f.d_nv12);
                f.d_nv12 = nullptr;
            }
        }

        // ── Phase 2: finish cur inference, GPU decode, submit nxt, NMS ───────

        if (primed) {
            if (engine.wait() != cudaSuccess) {
                std::cerr << "TRT sync failed\n"; continue;
            }

            // GPU decode: reset counts, launch decode kernels, then async copy
            // the tiny results — all on post_stream (non-blocking).
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
            // Async D2H of decode results on the same post_stream.
            // Pinned host memory allows true async DMA — CPU does not block.
            cudaMemcpyAsync(h_det_counts, d_det_counts_gpu,
                            (size_t)cur_sz * sizeof(int),
                            cudaMemcpyDeviceToHost, post_stream);
            cudaMemcpyAsync(h_dets_all, d_dets_gpu,
                            (size_t)cur_sz * MAX_DETS_PER_IMG * sizeof(Detection),
                            cudaMemcpyDeviceToHost, post_stream);

            // Submit next inference — TRT restarts immediately.
            // post_stream and engine.stream_ are both non-blocking-compatible:
            // they run independently on the GPU.
            if (have_nxt)
                engine.submit(d_input[nxt], d_output[nxt], (int)nxt_batch.size());

            // Wait for GPU decode + D2H (typically already done by now: ~0.1 ms).
            // This is NOT the null stream so it does NOT block TRT.
            cudaStreamSynchronize(post_stream);

            // NMS + stats — serial over the batch.
            // GPU already did the heavy work (confidence filter + box decode), so
            // each stream's NMS operates on only the handful of surviving boxes
            // (~10 faces).  That's a few microseconds; spawning a thread per frame
            // (std::async) costs more in thread-launch overhead than it saves, so
            // a plain loop is faster here.  All work runs concurrently with the
            // next TRT inference already submitted above.
            for (int b = 0; b < cur_sz; ++b) {
                const int sid = cur_batch[b].stream_id;
                const int cnt = std::min(h_det_counts[b], MAX_DETS_PER_IMG);
                const Detection* src = h_dets_all + (size_t)b * MAX_DETS_PER_IMG;
                std::vector<Detection> dets(src, src + cnt);

                fps_trackers[sid].tick();
                last_dets[sid] = nms(std::move(dets), cfg.nms_thresh);

                if (cur_batch[b].frame_num % (uint64_t)cfg.fps_window == 0)
                    std::cout << "[src=" << sid
                              << " frame=" << cur_batch[b].frame_num
                              << "] faces=" << last_dets[sid].size()
                              << " fps=" << std::fixed << std::setprecision(1)
                              << fps_trackers[sid].fps() << "\n";
            }

            // Display — bgr_mats already populated from Phase 1 GPU path.
            // cv::pollKey() instead of cv::waitKey(1): processes only currently
            // pending X11/Wayland events (non-blocking) — avoids the 5-25 ms
            // forced wait that waitKey(1) incurs in the hot path.
            // h_canvas[cur] holds this batch's GPU-composited frames (built in
            // the previous iteration's Phase 1, when cur was nxt).  Boxes are
            // from this same batch — image and overlay stay in sync.  We only
            // draw the lightweight vector overlay (boxes + labels) on the small
            // composed canvas, then show it.
            if (cfg.display) {
                for (int b = 0; b < cur_sz; ++b) {
                    const int sid = cur_batch[b].stream_id;
                    draw_tile(h_canvas[cur], sid, cols,
                              cfg.cell_w, cfg.cell_h,
                              cur_batch[b].src_w, cur_batch[b].src_h,
                              fps_trackers[sid].fps(), last_dets[sid]);
                }
                cv::imshow("rtsp_trt", h_canvas[cur]);
                if (cv::pollKey() == 27) break;
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

    // Drain final batch
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
        cudaMemcpyAsync(h_det_counts, d_det_counts_gpu, cur_sz * sizeof(int), cudaMemcpyDeviceToHost, post_stream);
        cudaMemcpyAsync(h_dets_all, d_dets_gpu, (size_t)cur_sz * MAX_DETS_PER_IMG * sizeof(Detection), cudaMemcpyDeviceToHost, post_stream);
        cudaStreamSynchronize(post_stream);
        for (int b = 0; b < cur_sz; ++b) {
            int sid = cur_batch[b].stream_id;
            int cnt = std::min(h_det_counts[b], MAX_DETS_PER_IMG);
            last_dets[sid] = nms(std::vector<Detection>(
                h_dets_all + b * MAX_DETS_PER_IMG,
                h_dets_all + b * MAX_DETS_PER_IMG + cnt), cfg.nms_thresh);
        }
    }

    std::cout << "Stopping...\n";
    for (auto& s : streams) s->stop();
    cv::destroyAllWindows();

    // Cleanup
    for (int i = 0; i < n; ++i)
        cudaStreamDestroy(pre_streams[i]);
    cudaStreamDestroy(dl_stream);
    cudaStreamDestroy(post_stream);
    for (int i = 0; i < 2; ++i) { cudaFree(d_input[i]); cudaFree(d_output[i]); }
    cudaFree(d_dets_gpu); cudaFree(d_det_counts_gpu);
    cudaFreeHost(h_det_counts); cudaFreeHost(h_dets_all);
    if (d_canvas) cudaFree(d_canvas);
    return 0;
}
