#include "stream.hpp"
#include <gst/video/video.h>
#include <cuda_runtime.h>
#include <iostream>

// GST_MAP_CUDA = (GST_MAP_FLAG_LAST << 1) = 1<<17 per GStreamer source.
// Not shipped as a dev header on standard Ubuntu.
#ifndef GST_MAP_CUDA
#define GST_MAP_CUDA ((GstMapFlags)(GST_MAP_FLAG_LAST << 1))
#endif

static bool uri_is_rtsp(const std::string& s) {
    return s.rfind("rtsp://", 0) == 0 || s.rfind("rtsps://", 0) == 0;
}

// ── Stream ────────────────────────────────────────────────────────────────────

Stream::Stream(int id, const std::string& uri)
    : id_(id), uri_(uri), is_file_(!uri_is_rtsp(uri)) {}

Stream::~Stream() { stop(); }

bool Stream::start() {
    pipeline_ = gst_pipeline_new(nullptr);

    appsink_ = gst_element_factory_make("appsink", "sink");
    if (!appsink_) {
        std::cerr << "[Stream " << id_ << "] appsink unavailable\n";
        return false;
    }

    GstCaps* sink_caps = gst_caps_from_string(
        "video/x-raw(memory:CUDAMemory),format=NV12");
    g_object_set(appsink_,
        "caps",         sink_caps,
        "sync",         FALSE,
        "max-buffers",  (guint)MAX_Q,
        "drop",         TRUE,
        "emit-signals", TRUE,
        nullptr);
    gst_caps_unref(sink_caps);

    GstAppSinkCallbacks cbs{};
    cbs.new_sample = on_new_sample;
    gst_app_sink_set_callbacks(GST_APP_SINK(appsink_), &cbs, this, nullptr);

    // Non-blocking copy stream — completely independent of TRT's engine stream.
    cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);

    if (is_file_) {
        filesrc_          = gst_element_factory_make("filesrc",   "src");
        GstElement* demux = gst_element_factory_make("qtdemux",   "demux");
        GstElement* parse = gst_element_factory_make("h264parse", "parse");
        GstElement* dec   = gst_element_factory_make("nvh264dec", "dec");
        if (!filesrc_ || !demux || !parse || !dec) {
            std::cerr << "[Stream " << id_
                      << "] Missing element (need filesrc qtdemux h264parse nvh264dec)\n";
            return false;
        }
        g_object_set(filesrc_, "location", uri_.c_str(), nullptr);
        g_object_set(dec, "num-output-surfaces", (guint)8, nullptr);

        gst_bin_add_many(GST_BIN(pipeline_),
            filesrc_, demux, parse, dec, appsink_, nullptr);
        if (!gst_element_link_many(parse, dec, appsink_, nullptr)) {
            std::cerr << "[Stream " << id_ << "] parse→dec→appsink link failed\n";
            return false;
        }
        gst_element_link(filesrc_, demux);
        g_signal_connect(demux, "pad-added", G_CALLBACK(on_qtdemux_pad), parse);

    } else {
        GstElement* src   = gst_element_factory_make("rtspsrc",      "src");
        GstElement* depay = gst_element_factory_make("rtph264depay", "depay");
        GstElement* parse = gst_element_factory_make("h264parse",    "parse");
        GstElement* dec   = gst_element_factory_make("nvh264dec",    "dec");
        if (!src || !depay || !parse || !dec) {
            std::cerr << "[Stream " << id_
                      << "] Missing element (need rtspsrc rtph264depay h264parse nvh264dec)\n";
            return false;
        }
        g_object_set(src,
            "location",        uri_.c_str(),
            "latency",         (guint)100,
            "protocols",       (guint)4,
            "drop-on-latency", TRUE,
            nullptr);
        g_object_set(dec, "num-output-surfaces", (guint)8, nullptr);

        gst_bin_add_many(GST_BIN(pipeline_),
            src, depay, parse, dec, appsink_, nullptr);
        if (!gst_element_link_many(depay, parse, dec, appsink_, nullptr)) {
            std::cerr << "[Stream " << id_ << "] depay→parse→dec→appsink link failed\n";
            return false;
        }
        g_signal_connect(src, "pad-added", G_CALLBACK(on_rtspsrc_pad), depay);
    }

    bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));

    running_ = true;
    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[Stream " << id_ << "] PLAYING failed\n";
        running_ = false;
        return false;
    }
    std::cout << "[Stream " << id_ << "] started: " << uri_ << "\n";
    return true;
}

void Stream::stop() {
    running_ = false;
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsink_  = nullptr;
        filesrc_  = nullptr;
    }
    if (bus_) { gst_object_unref(bus_); bus_ = nullptr; }

    if (copy_stream_) {
        cudaStreamSynchronize(copy_stream_);
        cudaStreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }

    // Drain queue and free the pool
    std::lock_guard<std::mutex> lk(mu_);
    while (!queue_.empty()) {
        pool_free_.push_back(queue_.front().d_nv12);
        queue_.pop();
    }
    for (auto* p : pool_free_) cudaFree(p);
    pool_free_.clear();
}

void Stream::poll_bus() {
    if (!bus_) return;
    GstMessage* msg = gst_bus_timed_pop_filtered(bus_, 0,
        GstMessageType(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!msg) return;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            if (is_file_ && filesrc_) {
                gst_element_seek_simple(pipeline_, GST_FORMAT_TIME,
                    GstSeekFlags(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
                std::cout << "[Stream " << id_ << "] looped\n";
            } else {
                std::cerr << "[Stream " << id_ << "] EOS\n";
                running_ = false;
            }
            break;
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr; gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::cerr << "[Stream " << id_ << "] ERROR: " << err->message;
            if (dbg) std::cerr << " | " << dbg;
            std::cerr << "\n";
            g_clear_error(&err); g_free(dbg);
            running_ = false;
            break;
        }
        default: break;
    }
    gst_message_unref(msg);
}

bool Stream::try_pop(Frame& out) {
    std::lock_guard<std::mutex> lk(mu_);
    if (queue_.empty()) return false;
    out = std::move(queue_.front());
    queue_.pop();
    return true;
}

void Stream::release_frame(uint8_t* d_nv12) {
    std::lock_guard<std::mutex> lk(mu_);
    pool_free_.push_back(d_nv12);
}

// ── pool ──────────────────────────────────────────────────────────────────────

void Stream::pool_init(int w, int h) {
    // Called under mu_ on the first decoded frame.
    pool_w_   = w;
    pool_h_   = h;
    pool_buf_ = (size_t)w * h * 3 / 2;   // packed NV12
    for (int i = 0; i < POOL_SZ; ++i) {
        uint8_t* p = nullptr;
        if (cudaMalloc(&p, pool_buf_) != cudaSuccess) {
            std::cerr << "[Stream " << id_ << "] cudaMalloc failed for NV12 pool\n";
            break;
        }
        pool_free_.push_back(p);
    }
}

uint8_t* Stream::pool_acquire() {
    // Called under mu_.
    if (pool_free_.empty()) return nullptr;
    uint8_t* p = pool_free_.back();
    pool_free_.pop_back();
    return p;
}

// ── pad callbacks ─────────────────────────────────────────────────────────────

void Stream::on_qtdemux_pad(GstElement* /*demux*/, GstPad* pad, gpointer data) {
    GstElement* parse = static_cast<GstElement*>(data);
    GstPad*     sink  = gst_element_get_static_pad(parse, "sink");
    if (!sink || gst_pad_is_linked(sink)) { if (sink) gst_object_unref(sink); return; }
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    if (caps) {
        if (g_str_has_prefix(gst_structure_get_name(gst_caps_get_structure(caps, 0)), "video/"))
            gst_pad_link(pad, sink);
        gst_caps_unref(caps);
    }
    gst_object_unref(sink);
}

void Stream::on_rtspsrc_pad(GstElement* /*src*/, GstPad* pad, gpointer data) {
    GstElement* depay = static_cast<GstElement*>(data);
    GstPad*     sink  = gst_element_get_static_pad(depay, "sink");
    if (!sink || gst_pad_is_linked(sink)) { if (sink) gst_object_unref(sink); return; }
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    bool ok = false;
    if (caps) {
        for (guint i = 0; i < gst_caps_get_size(caps); ++i) {
            GstStructure* s   = gst_caps_get_structure(caps, i);
            const gchar*  enc = gst_structure_get_string(s, "encoding-name");
            if (g_strcmp0(gst_structure_get_name(s), "application/x-rtp") == 0 &&
                enc && g_ascii_strcasecmp(enc, "H264") == 0) { ok = true; break; }
        }
        gst_caps_unref(caps);
    }
    if (ok) gst_pad_link(pad, sink);
    gst_object_unref(sink);
}

// ── appsink callback ──────────────────────────────────────────────────────────
//
// This replicates what nvinfer does internally:
//   1. Get CUDA device pointers to the decoder's NV12 output surface (via CUDAMemory map)
//   2. D2D-copy the frame into our own pool buffer (our CUDA context / TRT's context)
//      using a non-blocking copy stream — completely independent of TRT inference
//   3. Release the decoder surface immediately (unmap + unref)
//   4. Push the Frame carrying our own d_nv12 pointer
//
// Preprocessing then runs on our memory — no cross-context reads, no implicit
// synchronisation between GStreamer's CUcontext and TRT's primary context.

GstFlowReturn Stream::on_new_sample(GstAppSink* sink, gpointer data) {
    auto* self = static_cast<Stream*>(data);

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer* buf  = gst_sample_get_buffer(sample);
    GstCaps*   caps = gst_sample_get_caps(sample);
    if (!buf || !caps) { gst_sample_unref(sample); return GST_FLOW_OK; }

    GstVideoInfo vi;
    if (!gst_video_info_from_caps(&vi, caps)) {
        gst_sample_unref(sample); return GST_FLOW_OK;
    }

    // Map all NV12 planes as CUDA device pointers.
    // gst_video_frame_map handles multi-plane allocations correctly —
    // plane data[0] and data[1] are the real CUDA addresses for Y and UV
    // even if stored in separate memory objects.
    GstVideoFrame vf;
    if (!gst_video_frame_map(&vf, &vi, buf,
            (GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))) {
        std::cerr << "[Stream " << self->id_
                  << "] CUDA map failed — nvh264dec not outputting CUDAMemory\n";
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    const uint8_t* d_src_y  = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&vf, 0));
    const uint8_t* d_src_uv = static_cast<uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&vf, 1));
    const int sy  = GST_VIDEO_FRAME_PLANE_STRIDE(&vf, 0);
    const int suv = GST_VIDEO_FRAME_PLANE_STRIDE(&vf, 1);
    const int w   = GST_VIDEO_INFO_WIDTH(&vi);
    const int h   = GST_VIDEO_INFO_HEIGHT(&vi);

    {
        std::lock_guard<std::mutex> lk(self->mu_);

        // Lazy pool allocation on the first decoded frame (size unknown earlier).
        if (self->pool_free_.empty() && self->pool_w_ == 0)
            self->pool_init(w, h);

        uint8_t* d_dst = self->pool_acquire();
        if (d_dst) {
            uint8_t* d_dst_y  = d_dst;
            uint8_t* d_dst_uv = d_dst + w * h;

            // Destrided D2D copy on the non-blocking copy stream.
            // Strips decoder row-padding so our buffer is tightly packed (stride = w).
            // This is the same operation nvinfer performs internally before preprocessing.
            cudaMemcpy2DAsync(d_dst_y,  w, d_src_y,  sy,  w, h,   cudaMemcpyDeviceToDevice, self->copy_stream_);
            cudaMemcpy2DAsync(d_dst_uv, w, d_src_uv, suv, w, h/2, cudaMemcpyDeviceToDevice, self->copy_stream_);

            // Synchronise the copy stream so the transfer is complete before we
            // release the source surface.  ~20–50 µs for a 1080p NV12 frame.
            cudaStreamSynchronize(self->copy_stream_);

            if ((int)self->queue_.size() >= MAX_Q) {
                // Drop oldest — return its buffer to the pool.
                self->pool_free_.push_back(self->queue_.front().d_nv12);
                self->queue_.pop();
            }
            self->queue_.push({d_dst, w, h, self->id_, self->n_frames_++});
        }
        // Pool exhausted → drop silently (main thread is too slow).
    }

    // Surface can be released immediately — we've already finished the copy.
    gst_video_frame_unmap(&vf);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}
