#include "engine.hpp"
#include <NvInferRuntime.h>
#include <fstream>
#include <stdexcept>
#include <iostream>

void Engine::Logger::log(Severity sev, const char* msg) noexcept {
    if (sev <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

Engine::Engine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Engine file not found: " + path);
    std::string buf((std::istreambuf_iterator<char>(f)), {});

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");

    engine_ = runtime_->deserializeCudaEngine(buf.data(), buf.size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + path);

    ctx_ = engine_->createExecutionContext();
    if (!ctx_) throw std::runtime_error("createExecutionContext failed");

    cudaStreamCreate(&stream_);

    // TRT 10 requires the optimization profile to be activated before any
    // tensor addresses can be registered. A single-profile engine still needs
    // this explicit call or setTensorAddress silently has no effect.
    if (!ctx_->setOptimizationProfileAsync(0, stream_))
        throw std::runtime_error("setOptimizationProfileAsync(0) failed");
    cudaStreamSynchronize(stream_);

    // Print every tensor so mismatches are obvious
    int nb = engine_->getNbIOTensors();
    std::cout << "[Engine] " << nb << " IO tensors:\n";
    for (int i = 0; i < nb; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dims = engine_->getTensorShape(name);

        std::cout << "  [" << i << "] " << name
                  << "  mode=" << (int)mode
                  << "  dims=[";
        for (int d = 0; d < dims.nbDims; ++d)
            std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "," : "");
        std::cout << "]\n";

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            in_name_ = name;
            dynamic_  = (dims.d[0] == -1);
            if (dims.nbDims >= 4) {
                in_h_ = (int)dims.d[2];
                in_w_ = (int)dims.d[3];
            }
            if (dynamic_) {
                auto mx = engine_->getProfileShape(name, 0,
                              nvinfer1::OptProfileSelector::kMAX);
                max_batch_ = (int)mx.d[0];
            } else {
                max_batch_ = (int)dims.d[0];
            }
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            out_name_ = name;
            if (dims.nbDims >= 3) {
                // getProfileShape is INPUT-only in TRT 10 — output non-batch
                // dims are always static, read them directly.
                num_attrs_   = (int)dims.d[dims.nbDims - 2];
                num_anchors_ = (int)dims.d[dims.nbDims - 1];
            }
        }
    }

    if (in_name_.empty())  throw std::runtime_error("No INPUT tensor found in engine");
    if (out_name_.empty()) throw std::runtime_error("No OUTPUT tensor found in engine");

    std::cout << "[Engine] loaded: " << path << "\n"
              << "  input  = \"" << in_name_  << "\"  max_batch=" << max_batch_
              << "  H=" << in_h_ << " W=" << in_w_ << "  dynamic=" << dynamic_ << "\n"
              << "  output = \"" << out_name_ << "\"  attrs=" << num_attrs_
              << " anchors=" << num_anchors_ << "\n";
}

Engine::~Engine() {
    cudaStreamDestroy(stream_);
    delete ctx_;
    delete engine_;
    delete runtime_;
}

bool Engine::submit(void* d_input, void* d_output, int batch) {
    if (dynamic_) {
        nvinfer1::Dims4 shape{batch, 3, in_h_, in_w_};
        if (!ctx_->setInputShape(in_name_.c_str(), shape)) {
            std::cerr << "[Engine] setInputShape failed for batch=" << batch << "\n";
            return false;
        }
    }
    if (!ctx_->setTensorAddress(in_name_.c_str(), d_input)) {
        std::cerr << "[Engine] setTensorAddress(input) failed\n";
        return false;
    }
    if (!ctx_->setOutputTensorAddress(out_name_.c_str(), d_output)) {
        std::cerr << "[Engine] setOutputTensorAddress failed\n";
        return false;
    }
    if (!ctx_->enqueueV3(stream_)) {
        std::cerr << "[Engine] enqueueV3 failed\n";
        return false;
    }
    return true;
}

cudaError_t Engine::wait() {
    return cudaStreamSynchronize(stream_);
}

bool Engine::infer(void* d_input, void* d_output, int batch) {
    return submit(d_input, d_output, batch) && (wait() == cudaSuccess);
}
