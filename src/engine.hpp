#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>

class Engine {
public:
    explicit Engine(const std::string& path);
    ~Engine();

    // Blocking inference (enqueue + wait).  For sequential loops.
    bool infer(void* d_input, void* d_output, int batch);

    // Async inference: enqueue on the engine's CUDA stream and return immediately.
    // The caller must call wait() before reading d_output or submitting again.
    bool submit(void* d_input, void* d_output, int batch);

    // Wait for the previously submitted inference to finish.
    cudaError_t wait();

    // The CUDA stream used for inference — useful to chain async host-side work.
    cudaStream_t cuda_stream() const { return stream_; }

    int max_batch()    const { return max_batch_; }
    int input_h()      const { return in_h_; }
    int input_w()      const { return in_w_; }
    int num_attrs()    const { return num_attrs_; }
    int num_anchors()  const { return num_anchors_; }

private:
    struct Logger : nvinfer1::ILogger {
        void log(Severity sev, const char* msg) noexcept override;
    } logger_;

    nvinfer1::IRuntime*          runtime_  = nullptr;
    nvinfer1::ICudaEngine*       engine_   = nullptr;
    nvinfer1::IExecutionContext* ctx_      = nullptr;
    cudaStream_t                 stream_   = nullptr;

    std::string in_name_, out_name_;
    int         max_batch_   = 1;
    int         in_h_        = 640;
    int         in_w_        = 640;
    int         num_attrs_   = 5;
    int         num_anchors_ = 8400;
    bool        dynamic_     = false;
};
