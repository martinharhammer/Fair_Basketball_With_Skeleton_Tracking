from __future__ import annotations
from time import time
from typing import Optional

class ProgressLogger:
    def __init__(
        self,
        prefix: str,
        total: Optional[int] = None,
        log_every: Optional[int] = 50,
        log_every_s: Optional[float] = None,
    ):
        self.prefix = prefix
        self.total = total if (isinstance(total, int) and total > 0) else None
        self.log_every = max(1, int(log_every)) if log_every else None
        self.log_every_s = float(log_every_s) if log_every_s else None

        self._count = 0
        self._next_count = (self._count + self.log_every) if self.log_every else None
        self._t0 = time()
        self._next_time = (self._t0 + self.log_every_s) if self.log_every_s else None

    def tick(self, n: int = 1) -> None:
        self._count += n
        should = False

        if self.log_every and self._count >= (self._next_count or 0):
            should = True
            self._next_count = self._count + self.log_every

        now = time()
        if self.log_every_s and now >= (self._next_time or 0):
            should = True
            self._next_time = now + self.log_every_s

        if should:
            if self.total:
                print(f"[{self.prefix}] Processed {self._count}/{self.total}")
            else:
                print(f"[{self.prefix}] Processed {self._count}")

    def done(self, suffix: Optional[str] = None) -> None:
        if suffix:
            print(f"[{self.prefix}] {suffix}")

