"""
Rate Limiter
=============
Client-side token + request rate limiter for Groq free-tier.
Tracks usage over a rolling 60-second window.
"""
from __future__ import annotations

import time
import threading
from collections import deque


class RateLimiter:
    """
    Sliding-window rate limiter tracking both RPM and TPM.
    Thread-safe via lock.
    """

    def __init__(self, rpm: int = 28, tpm: int = 7500, window: float = 60.0):
        self.rpm = rpm
        self.tpm = tpm
        self.window = window
        self._req_times: deque[float] = deque()
        self._tok_log: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = threading.RLock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._tok_log and self._tok_log[0][0] < cutoff:
            self._tok_log.popleft()

    def tokens_used(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return sum(t for _, t in self._tok_log)

    def wait_if_needed(self, estimated_tokens: int = 300) -> None:
        """Block until there is capacity for the request."""
        while True:
            now = time.monotonic()
            with self._lock:
                self._prune(now)
                req_ok = len(self._req_times) < self.rpm
                tok_ok = self.tokens_used() + estimated_tokens <= self.tpm
                if req_ok and tok_ok:
                    return
                # Compute sleep time
                if self._req_times and not req_ok:
                    sleep = self.window - (now - self._req_times[0]) + 0.1
                elif self._tok_log and not tok_ok:
                    sleep = self.window - (now - self._tok_log[0][0]) + 0.1
                else:
                    sleep = 1.0
            time.sleep(max(0.1, sleep))

    def record(self, tokens_used: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._req_times.append(now)
            self._tok_log.append((now, tokens_used))


# Singleton
_limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    return _limiter
