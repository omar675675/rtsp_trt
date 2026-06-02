"""
rtsp_trt Python wrapper.

Launches the compiled C++ binary and exposes a clean Python API.

Usage — run directly:
    python3 pipeline.py [config.yaml]

Usage — import:
    from pipeline import Pipeline, Stats

    def on_stats(s: Stats):
        print(f"stream {s.stream_id}  fps={s.fps:.1f}  faces={s.faces}")

    p = Pipeline("config.yaml", on_stats=on_stats)
    p.run()          # blocking — Ctrl+C stops cleanly

    # or non-blocking:
    p.start()
    ...
    p.stop()
"""

import os
import re
import sys
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── types ─────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    stream_id: int
    frame_num: int
    faces:     int
    fps:       float

# Stats lines emitted by the C++ binary:
# [src=0 frame=30] faces=2 fps=29.8
_STATS_RE = re.compile(
    r"\[src=(\d+)\s+frame=(\d+)\]\s+faces=(\d+)\s+fps=([\d.]+)"
)

# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Thin wrapper around the rtsp_trt C++ binary.

    Args:
        config:   path to config.yaml
        on_stats: optional callback called for each stats line the binary prints;
                  receives a Stats dataclass, called from a background thread
        binary:   path to the compiled binary (defaults to build/rtsp_trt next to
                  this file)
    """

    DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

    def __init__(
        self,
        config: str | Path | None = None,
        on_stats: Optional[Callable[[Stats], None]] = None,
        binary: Optional[str | Path] = None,
    ):
        self.config   = Path(config).resolve() if config else self.DEFAULT_CONFIG
        self.on_stats = on_stats
        self.binary   = Path(binary) if binary else \
                        Path(__file__).parent / "build" / "rtsp_trt"

        self._proc:   Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._lock    = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the pipeline in the background (non-blocking)."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                raise RuntimeError("Pipeline is already running")
            self._launch()

    def stop(self, timeout: float = 5.0) -> None:
        """Send SIGINT to the binary and wait for it to exit cleanly."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
        if self._reader:
            self._reader.join(timeout=2.0)

    def run(self) -> None:
        """Launch the pipeline and block until it stops or Ctrl+C is pressed."""
        self.start()
        try:
            self._proc.wait()
        except KeyboardInterrupt:
            self.stop()
        finally:
            if self._reader:
                self._reader.join(timeout=5.0)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── internals ─────────────────────────────────────────────────────────

    def _launch(self) -> None:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"Binary not found: {self.binary}\n"
                f"  Run:  cd {self.binary.parent.parent} && "
                f"mkdir -p build && cd build && cmake .. && make -j$(nproc)"
            )
        if not self.config.exists():
            raise FileNotFoundError(f"Config not found: {self.config}")

        env = os.environ.copy()
        env["RTSP_TRT_CONFIG"] = str(self.config)

        self._proc = subprocess.Popen(
            [str(self.binary), str(self.config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → one stream to parse
            text=True,
            bufsize=1,                  # line-buffered
            env=env,
        )
        self._reader = threading.Thread(
            target=self._read_output, daemon=True, name="rtsp_trt_reader"
        )
        self._reader.start()

    def _read_output(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)         # mirror to our stdout

            if self.on_stats:
                m = _STATS_RE.search(line)
                if m:
                    try:
                        self.on_stats(Stats(
                            stream_id = int(m.group(1)),
                            frame_num = int(m.group(2)),
                            faces     = int(m.group(3)),
                            fps       = float(m.group(4)),
                        ))
                    except Exception as e:
                        print(f"[pipeline.py] on_stats error: {e}", file=sys.stderr)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = Pipeline()
    try:
        p.run()
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
