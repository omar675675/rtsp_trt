import sys
import queue
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rtsp_trt import Pipeline, StatsEvent, DetectionEvent

CONFIG = Path(__file__).parent / "config.yaml"
LABEL  = "face"   # change to match whatever model you're running

# Per-stream buffer: stream_id → [frame_num, frame_ndarray, [(x,y,w,h,conf,tid)]]
_lock    = threading.Lock()
_pending = {}
_display_q: queue.Queue = queue.Queue(maxsize=16)


def on_stats(s: StatsEvent):
    print(f"[src={s.stream_id} frame={s.frame_num}] objects={s.faces} fps={s.fps:.1f}")


def on_detection(d: DetectionEvent):
    with _lock:
        prev = _pending.get(d.stream_id)

        if prev is not None and prev[0] != d.frame_num:
            # Frame boundary — push completed frame to display thread
            try:
                _display_q.put_nowait((d.stream_id, prev[0], prev[1], prev[2]))
            except queue.Full:
                pass  # display can't keep up — drop
            prev = None

        if prev is None:
            _pending[d.stream_id] = [d.frame_num, np.array(d.frame), []]
            prev = _pending[d.stream_id]

        if d.conf > 0:
            prev[2].append((d.x, d.y, d.w, d.h, d.conf, d.track_id))


def draw(stream_id, frame_bgr, boxes):
    img = frame_bgr.copy()
    for x, y, w, h, conf, tid in boxes:
        x1 = max(0,           int(x))
        y1 = max(0,           int(y))
        x2 = min(img.shape[1], int(x + w))
        y2 = min(img.shape[0], int(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{LABEL} #{tid}  {conf:.2f}",
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imshow(f"stream {stream_id}", img)


p = Pipeline(str(CONFIG))
p.set_on_stats(on_stats)
p.set_on_detection(on_detection)
p.start()

print("Running — press Q to quit")
try:
    while p.running:
        try:
            sid, fnum, frame, boxes = _display_q.get(timeout=0.05)
            draw(sid, frame, boxes)
        except queue.Empty:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    p.stop()
    cv2.destroyAllWindows()
