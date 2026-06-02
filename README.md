# rtsp_trt

Multi-stream face detection pipeline built on **TensorRT 10 + GStreamer + CUDA**.  
A container-free reimplementation of the DeepStream pipeline at `../deepstream/`, with native RTSP support and no NVIDIA DeepStream dependency.

---

## What it does

- Accepts any number of **file** (MP4/H.264) or **RTSP** sources defined in `config.yaml`
- Decodes all streams in parallel via GStreamer (`decodebin` auto-selects HW/SW decoder)
- Batches one frame per stream, preprocesses on GPU (letterbox resize + BGR→RGB + normalize), runs a single **YOLOv12m-face** TensorRT FP16 inference call on the batch
- Decodes boxes + runs CPU NMS per stream
- **Headless mode** (`display: false`): prints per-stream FPS/face-count stats, no window
- **Display mode** (`display: true`): tiled OpenCV window with green bounding boxes and per-stream FPS/face-count overlay

File sources loop forever (EOS seek). RTSP sources stop on EOS.

---

## System requirements

| Requirement | Version used | Notes |
|---|---|---|
| OS | Ubuntu 24.04 | |
| GPU | RTX 4060 Laptop (SM 8.9) | CMake CUDA arch is set to `89` |
| CUDA | 12.6 | headers at `/usr/local/cuda` |
| **TensorRT** | **10.16.1** | Must be TRT 10 — engine was built with `--fp16` which requires TRT 10. TRT 11 removed the flag. |
| GStreamer | 1.24.2 | `gstreamer1.0-*` packages |
| OpenCV | 4.6.0 | `libopencv-dev` |
| yaml-cpp | 0.8.0 | `libyaml-cpp-dev` |

> **TRT version is critical.** The `.engine` file was compiled with TRT 10.  
> Engines are not portable across major TRT versions. If you upgrade TRT, rebuild the engine (see below).

---

## Repository layout

```
rtsp_trt/
├── CMakeLists.txt          Build definition
├── config.yaml             Runtime config — single source of truth
├── pipeline.py             Python wrapper (import or run directly)
├── models/
│   ├── yolov12m-face.pt    PyTorch weights
│   ├── yolov12m-face.onnx  ONNX export (dynamic batch, opset 17)
│   └── yolov12m-face.engine TRT FP16 engine (TRT 10, SM 8.9, max batch 10)
├── streams/
│   └── test1-8.mp4         Test H.264 files
└── src/
    ├── main.cpp            Orchestration: config → streams → inference loop → display
    ├── engine.hpp/cpp      TensorRT 10 wrapper (deserialize, dynamic batch, enqueueV3)
    ├── stream.hpp/cpp      Per-source GStreamer pipeline (file + RTSP, decodebin, appsink)
    ├── preprocess.cuh/cu   CUDA kernel: letterbox resize + BGR→RGB + normalize CHW float32
    ├── postprocess.hpp/cpp Box decode (attr-major tensor layout) + greedy IoU NMS
    └── display.hpp/cpp     OpenCV bounding box draw + tiled canvas composer
```

---

## Build

```bash
cd rtsp_trt
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The binary lands at `build/rtsp_trt`.

**If your GPU is not SM 8.9** (RTX 40xx), change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`:
- RTX 30xx → `86`
- RTX 20xx → `75`
- A100 → `80`

---

## Rebuilding the TRT engine

Required whenever TRT major version changes or you switch GPU generation.

```bash
trtexec \
  --onnx=models/yolov12m-face.onnx \
  --saveEngine=models/yolov12m-face.engine \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:10x3x640x640 \
  --fp16
```

`--fp16` requires TRT 10. TRT 11 removed this flag (replaced with `--precisionFlags=fp16`).  
The opt shape of `4` means batch=4 will be fastest; min=1 and max=10 define the valid range.

---

## Configuration (`config.yaml`)

All tunables live here. The C++ binary and Python wrapper both read this file.  
**Never edit `src/` to change thresholds or paths — edit this file.**

```yaml
display: false          # true = tiled OpenCV window, false = headless (faster)

model:
  engine: /abs/path/to/yolov12m-face.engine
  input_width: 640
  input_height: 640
  engine_max_batch: 10  # must match --maxShapes batch used in trtexec

detection:
  conf_threshold: 0.50  # pre-NMS score filter
  nms_threshold: 0.45   # IoU threshold for greedy NMS

streams:                # add/remove freely — batch-size adapts automatically
  - /path/to/file.mp4
  - rtsp://user:pass@ip/stream

hw_decode: true         # hint to decodebin to prefer nvh264dec over avdec_h264

tiler:                  # only used when display: true
  columns: 2
  cell_width: 640
  cell_height: 360

fps_window: 30          # frames averaged for the FPS readout
```

---

## Running

### Direct binary

```bash
./build/rtsp_trt               # uses config.yaml in cwd
./build/rtsp_trt /path/to/config.yaml
RTSP_TRT_CONFIG=/path/to/config.yaml ./build/rtsp_trt
```

### Python wrapper

```bash
python pipeline.py             # uses config.yaml next to pipeline.py
```

```python
from pipeline import Pipeline, Stats

def on_stats(s: Stats):
    print(f"stream {s.stream_id}  fps={s.fps:.1f}  faces={s.faces}")

# Blocking
Pipeline(on_stats=on_stats).run()

# Non-blocking (embed in larger app)
p = Pipeline(on_stats=on_stats)
p.start()
# ... other work ...
p.stop()
```

`Stats` fields: `stream_id: int`, `frame_num: int`, `faces: int`, `fps: float`.  
The `on_stats` callback fires from a background thread on every stats line the binary prints.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  GStreamer (one pipeline per source)                             │
│                                                                  │
│  filesrc/rtspsrc → decodebin → videoconvert → BGR appsink       │
│  filesrc/rtspsrc → decodebin → videoconvert → BGR appsink       │
│  ...                                                             │
└──────────────────────────────┬───────────────────────────────────┘
                               │ cv::Mat BGR frames (try_pop, non-blocking)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Inference loop (main thread)                                    │
│                                                                  │
│  1. Collect one frame per stream (non-blocking)                  │
│  2. CUDA preprocess each frame into batch slot:                  │
│     letterbox resize → BGR→RGB → normalize [0,1] → CHW float32  │
│  3. TRT enqueueV3: [batch, 3, 640, 640] → [batch, 5, 8400]      │
│  4. cudaMemcpy output to host                                    │
│  5. Decode boxes + NMS per stream                                │
│  6. Print stats / draw OpenCV overlay                            │
└──────────────────────────────────────────────────────────────────┘
```

### Output tensor layout

The TRT engine outputs `[batch, 5, 8400]` in **attribute-major** (channel-first) order:

```
output[attr * 8400 + anchor_idx]   where attr ∈ {cx, cy, w, h, score}
```

This is the same layout as the original `nvdsparse_yoloface.cpp` parser. Boxes are in **640×640 network-input coordinates**; the postprocessor applies the inverse letterbox transform to get original frame coordinates.

### Letterbox geometry

Stored in `LetterboxInfo { scale, pad_x, pad_y, src_w, src_h }` per frame, filled during preprocessing. Inverse transform in `postprocess.cpp`:

```
frame_x = (network_cx ± w/2 - pad_x) / scale
```

---

## Key implementation notes

### TRT 10 quirks (engine.cpp)

- `getProfileShape()` is **input-only** in TRT 10. Calling it on the output tensor returns garbage. Output non-batch dims (`5`, `8400`) are read directly from `getTensorShape()`.
- `setOptimizationProfileAsync(0, stream)` must be called once after context creation before any `setInputShape` or address calls.
- Output tensor uses `setOutputTensorAddress()` (not the generic `setTensorAddress`) because TRT 10 routes profile-oblivious output tensors through a different internal slot.
- Output tensor `output0` has dims `[-1, 5, 8400]` (dynamic batch). Despite this, TRT 10 internally classifies it as "profile-oblivious" — `setOutputTensorAddress` handles this correctly.

### GStreamer dynamic pads (stream.cpp)

Both `decodebin` (for file sources) and `rtspsrc` (for RTSP) expose dynamic pads. The `on_pad_added` callback fires when the video pad is available and links it to `videoconvert`. Only pads with MIME type starting `video/` are linked to avoid accidentally linking audio pads.

RTSP sources: `rtspsrc` is used directly (not `uridecodebin`) for control over `latency`, `protocols=TCP`, and `drop-on-latency`.

File sources loop via EOS seek in `poll_bus()`: when `GST_MESSAGE_EOS` arrives on the bus, `gst_element_seek_simple(..., 0)` restarts the file. `poll_bus()` is called from the main loop on every iteration (non-blocking poll with timeout=0).

### CUDA preprocessing (preprocess.cu)

One CUDA kernel per frame, one thread per output pixel. For each `(x, y)` in the 640×640 output:
1. Map back to source coordinates via `(dst - pad) / scale`
2. Nearest-neighbour sample from BGR source
3. Convert to RGB, normalize, write to CHW slot in the batch input buffer

Temporary source frames are uploaded with `cudaMallocAsync`/`cudaMemcpyAsync` and freed after the kernel returns (async on the preprocessing stream).

---

## Current status / known issues

| Issue | Status |
|---|---|
| TRT 10 `setOutputTensorAddress` + profile-oblivious tensor | Fix applied in `engine.cpp` — needs verification |
| Output dims detection (`attrs=0 anchors=-16777216`) | Fixed: no longer calls `getProfileShape` for output |
| File source looping | Working via `poll_bus()` EOS seek |
| RTSP streams | Untested — code path is in `stream.cpp`, uses `rtspsrc` |
| Display mode | Untested — OpenCV tiled window code in `display.cpp` |
| Batching when not all streams have a frame | Works — infers on available subset, updates only those stream slots |

---

## Regenerating the ONNX (if needed)

Requires `ultralytics` installed:

```bash
python3 - <<'EOF'
from ultralytics import YOLO
YOLO("models/yolov12m-face.pt").export(
    format="onnx", imgsz=640, dynamic=True, opset=17, simplify=True
)
EOF
```

Output lands next to the `.pt` file. Move to `models/` if needed, then re-run `trtexec`.

---

## Relationship to `../deepstream/`

| | `deepstream/` | `rtsp_trt/` |
|---|---|---|
| Requires container | Yes (NVIDIA DeepStream 7.1) | No |
| Inference | `nvinfer` element (DeepStream) | Direct TRT 10 C++ API |
| Box decode | `libnvdsparse_yoloface.so` (custom nvinfer parser) | `postprocess.cpp` (same logic) |
| Multi-stream mux | `nvstreammux` | Per-stream appsink + batch accumulator |
| Display | `nvmultistreamtiler + nvdsosd + nveglglessink` | OpenCV tiled imshow |
| RTSP support | Yes (nvstreammux) | Yes (rtspsrc) |
| Python entry point | `apps/pipeline.py` (GStreamer/pyds) | `pipeline.py` (subprocess wrapper) |
