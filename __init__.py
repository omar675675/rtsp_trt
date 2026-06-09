"""
rtsp_trt — native Python extension.

Import the compiled .so directly; no subprocess, no pipes.

Usage:
    from rtsp_trt import Pipeline, StatsEvent, DetectionEvent
    import numpy as np

    def on_stats(s: StatsEvent):
        print(f"stream {s.stream_id}  fps={s.fps:.1f}  faces={s.faces}")

    def on_detection(d: DetectionEvent):
        # d.crop  → (112, 112, 3) uint8 BGR numpy array
        # d.box   → (x, y, w, h) tuple in original frame coordinates
        print(f"stream {d.stream_id}  frame {d.frame_num}  conf={d.conf:.2f}  box={d.box}")

    p = Pipeline("config.yaml")
    p.set_on_stats(on_stats)
    p.set_on_detection(on_detection)
    p.run()          # blocking — Ctrl+C stops cleanly

    # or non-blocking:
    p.start()
    ...
    p.stop()
"""

from ._rtsp_trt import Pipeline, StatsEvent, DetectionEvent

__all__ = ["Pipeline", "StatsEvent", "DetectionEvent"]
