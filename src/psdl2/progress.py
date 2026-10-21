"""Progress JSON lines on stderr, same format as DL1: {"event": "progress", "phase": ..., ...}."""
from __future__ import annotations

import json
import sys
import time


def stderr_sink(record):
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


class Progress:
    """No sink means no output. due() is True at most once per interval, so batch lines do not flood."""

    def __init__(self, sink=None, interval=10.0):
        self.sink, self.interval, self._last = sink, interval, time.monotonic()

    def emit(self, phase, **record):
        if self.sink is not None:
            self.sink({'event': 'progress', 'phase': phase, **record})

    def due(self):
        now = time.monotonic()
        if self.sink is None or now - self._last < self.interval:
            return False
        self._last = now
        return True
