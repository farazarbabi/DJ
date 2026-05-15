"""Single-line progress bar for CLI batch jobs, backed by tqdm."""

from __future__ import annotations

import sys

from tqdm import tqdm


class ProgressBar:
    """Thin wrapper over tqdm that keeps a stable API for batch jobs.

    Stays on a single line, overwrites in place, and surfaces inline counts
    (cached/analyzed/etc.) via the postfix area instead of separate log lines.
    """

    def __init__(self, total: int, *, label: str, enabled: bool = True, width: int = 28) -> None:
        self.total = max(0, total)
        self.label = label
        self.enabled = enabled
        self._finished = False
        self._bar: tqdm | None = None
        if self.enabled and self.total > 0:
            self._bar = tqdm(
                total=self.total,
                desc=label,
                unit="track",
                leave=True,
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.1,
            )

    def update(self, current: int, status: str = "", **counts: int) -> None:
        if not self.enabled or self._bar is None:
            return
        delta = current - self._bar.n
        if delta > 0:
            self._bar.update(delta)
        elif delta < 0:
            self._bar.n = current
            self._bar.refresh()
        postfix_parts: list[str] = []
        if counts:
            postfix_parts.extend(f"{k}={v}" for k, v in counts.items())
        if status:
            postfix_parts.append(status[:60])
        if postfix_parts:
            self._bar.set_postfix_str(" ".join(postfix_parts), refresh=False)
            self._bar.refresh()

    def finish(self, status: str = "complete") -> None:
        if self._finished:
            return
        self._finished = True
        if self._bar is None:
            return
        if status and status != "complete":
            self._bar.set_postfix_str(status)
        self._bar.close()
        self._bar = None
