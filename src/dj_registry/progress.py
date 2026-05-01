"""Small stderr progress bar used by registry batch workflows."""

from __future__ import annotations

import sys


class ProgressBar:
    """Dependency-free progress bar for CLI batch jobs."""

    def __init__(self, total: int, *, label: str, enabled: bool = True, width: int = 28) -> None:
        self.total = max(0, total)
        self.label = label
        self.enabled = enabled
        self.width = width
        self._finished = False

    def update(self, current: int, status: str = "", **counts: int) -> None:
        if not self.enabled:
            return
        total = max(1, self.total)
        ratio = min(1.0, max(0.0, current / total))
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        count_text = " ".join(f"{key}={value}" for key, value in counts.items())
        suffix = " ".join(part for part in (count_text, status[:60]) if part)
        sys.stderr.write(f"\r{self.label} [{bar}] {current}/{self.total} {suffix}".rstrip())
        sys.stderr.flush()

    def finish(self, status: str = "complete") -> None:
        if not self.enabled or self._finished:
            return
        sys.stderr.write(f"\n{self.label}: {status}\n")
        sys.stderr.flush()
        self._finished = True
