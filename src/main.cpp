#include "pipeline_core.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const char* env_cfg = std::getenv("RTSP_TRT_CONFIG");
    const std::string cfg_path = argc > 1 ? argv[1]
                               : (env_cfg ? env_cfg : "config.yaml");
    try {
        Pipeline p(cfg_path);
        p.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
