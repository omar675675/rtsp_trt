#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <cuda_runtime.h>
#include <queue>
#include <vector>
#include <mutex>
#include <atomic>
#include <string>
#include <cstdint>

// A decoded frame in our own CUDA device memory.
//
// d_nv12   — packed NV12: Y plane [src_h × src_w] bytes then UV plane
//            [src_h/2 × src_w] bytes.  Allocated from Stream's pool;
//            MUST be returned via Stream::release_frame() once the
//            preprocessing kernel has been synchronised.
//
// This matches what nvinfer does internally: it D2D-copies the nvstreammux
// frame into its own staging buffer (same CUDA context as TRT) before running
// the preprocessing kernel — eliminating cross-context read overhead.
struct Frame {
    uint8_t* d_nv12;    // our CUDA context memory, packed NV12
    int      src_w, src_h;
    int      stream_id;
    uint64_t frame_num;
};

class Stream {
public:
    Stream(int id, const std::string& uri);
    ~Stream();

    bool start();
    void stop();

    // Non-blocking.  Returns false when no frame is ready.
    bool try_pop(Frame& out);

    // Return a buffer to the pool once the preprocessing kernel has synced.
    void release_frame(uint8_t* d_nv12);

    // Non-blocking bus poll — call every main-loop iteration.
    void poll_bus();

    bool running() const { return running_; }
    int  id()      const { return id_; }

private:
    static GstFlowReturn on_new_sample(GstAppSink*, gpointer);
    static void on_qtdemux_pad(GstElement*, GstPad*, gpointer);
    static void on_rtspsrc_pad(GstElement*, GstPad*, gpointer);

    // Pool of pre-allocated NV12 device buffers (our CUDA context).
    // Sized so the decoder never stalls: queue depth + one being preprocessed
    // by main + one being written by the callback.
    static constexpr int MAX_Q   = 4;
    static constexpr int POOL_SZ = MAX_Q + 2;

    void     pool_init(int w, int h);  // lazy, called on first frame
    uint8_t* pool_acquire();
    // pool_release is exposed as release_frame() in the public API

    int    pool_w_   = 0, pool_h_ = 0;
    size_t pool_buf_ = 0;
    std::vector<uint8_t*> pool_free_;

    // Per-stream copy stream — cudaStreamNonBlocking so it never serialises
    // with TRT's engine stream or the null stream.
    cudaStream_t copy_stream_ = nullptr;

    int         id_;
    std::string uri_;
    bool        is_file_;


    GstElement* pipeline_ = nullptr;
    GstElement* appsink_  = nullptr;
    GstElement* filesrc_  = nullptr;
    GstBus*     bus_      = nullptr;

    std::queue<Frame> queue_;
    std::mutex        mu_;
    std::atomic<bool> running_{false};
    uint64_t          n_frames_{0};
};
