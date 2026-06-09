#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pipeline_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_rtsp_trt, m) {
    m.doc() = "Native rtsp_trt pipeline: GStreamer decode + TensorRT inference + GPU crop";

    py::class_<Pipeline::StatsEvent>(m, "StatsEvent")
        .def_readonly("stream_id", &Pipeline::StatsEvent::stream_id)
        .def_readonly("frame_num", &Pipeline::StatsEvent::frame_num)
        .def_readonly("faces",     &Pipeline::StatsEvent::faces)
        .def_readonly("fps",       &Pipeline::StatsEvent::fps)
        .def("__repr__", [](const Pipeline::StatsEvent& e) {
            return "[src=" + std::to_string(e.stream_id) +
                   " frame=" + std::to_string(e.frame_num) +
                   "] faces=" + std::to_string(e.faces) +
                   " fps=" + std::to_string(e.fps);
        });

    py::class_<Pipeline::DetectionEvent>(m, "DetectionEvent")
        .def_readonly("stream_id", &Pipeline::DetectionEvent::stream_id)
        .def_readonly("frame_num", &Pipeline::DetectionEvent::frame_num)
        .def_readonly("x",         &Pipeline::DetectionEvent::x)
        .def_readonly("y",         &Pipeline::DetectionEvent::y)
        .def_readonly("w",         &Pipeline::DetectionEvent::w)
        .def_readonly("h",         &Pipeline::DetectionEvent::h)
        .def_readonly("conf",      &Pipeline::DetectionEvent::conf)
        .def_readonly("track_id",  &Pipeline::DetectionEvent::track_id)
        .def_readonly("frame_w",   &Pipeline::DetectionEvent::frame_w)
        .def_readonly("frame_h",   &Pipeline::DetectionEvent::frame_h)
        .def_property_readonly("box", [](const Pipeline::DetectionEvent& e) {
            return py::make_tuple(e.x, e.y, e.w, e.h);
        })
        .def_property_readonly("frame", [](const Pipeline::DetectionEvent& e) {
            // Zero-copy: the numpy array keeps the shared_ptr alive via a capsule.
            using Buf = std::shared_ptr<const std::vector<uint8_t>>;
            auto* owner = new Buf(e.frame_bgr);
            py::capsule cap(owner, [](void* p){ delete static_cast<Buf*>(p); });
            return py::array_t<uint8_t>(
                {e.frame_h, e.frame_w, 3},
                (*e.frame_bgr).data(),
                cap);
        })
        .def("__repr__", [](const Pipeline::DetectionEvent& e) {
            return "[src=" + std::to_string(e.stream_id) +
                   " frame=" + std::to_string(e.frame_num) +
                   " track=" + std::to_string(e.track_id) +
                   "] conf=" + std::to_string(e.conf) +
                   " box=(" + std::to_string((int)e.x) + "," +
                               std::to_string((int)e.y) + "," +
                               std::to_string((int)e.w) + "," +
                               std::to_string((int)e.h) + ")";
        });

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<const std::string&>(), py::arg("config"),
             "Create a pipeline from a YAML config file path.")

        .def("set_on_stats",
            [](Pipeline& p, py::object cb) {
                if (cb.is_none()) {
                    p.set_on_stats(nullptr);
                    return;
                }
                // Keep a reference to the callable alive inside the lambda.
                py::object ref = cb;
                p.set_on_stats([ref](const Pipeline::StatsEvent& e) {
                    py::gil_scoped_acquire gil;
                    ref(e);
                });
            },
            py::arg("callback"),
            "Set a callback(StatsEvent) called every fps_window frames per stream.")

        .def("set_on_detection",
            [](Pipeline& p, py::object cb) {
                if (cb.is_none()) {
                    p.set_on_detection(nullptr);
                    return;
                }
                py::object ref = cb;
                p.set_on_detection([ref](const Pipeline::DetectionEvent& e) {
                    py::gil_scoped_acquire gil;
                    ref(e);
                });
            },
            py::arg("callback"),
            "Set a callback(DetectionEvent) for each detected face with a 112×112 BGR crop.")

        .def("start", &Pipeline::start,
             "Launch the pipeline in a background thread (non-blocking).")

        .def("stop", &Pipeline::stop, py::arg("timeout_sec") = 5.0,
             "Signal the pipeline to stop and wait up to timeout_sec seconds.")

        .def("run", &Pipeline::run,
             py::call_guard<py::gil_scoped_release>(),
             "Run the pipeline in the calling thread (blocking). "
             "Returns when the stream ends or Ctrl+C is pressed.")

        .def_property_readonly("running", &Pipeline::running,
             "True while the pipeline loop is executing.");
}
