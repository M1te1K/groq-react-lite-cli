"""Общий дедлайн, включая синхронные SDK, повторы и медленные ответы."""

import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class RequestExpired(Exception):
    pass


class Budget:
    def __init__(self, seconds: float, on_retry: Callable[[str, int], None] | None = None):
        self.expires_at = time.monotonic() + seconds
        self.on_retry = on_retry

    def retry(self, service: str, attempt: int) -> None:
        self.remaining()
        if self.on_retry:
            self.on_retry(service, attempt)

    def remaining(self) -> float:
        remaining = self.expires_at - time.monotonic()
        if remaining <= 0:
            raise RequestExpired
        return remaining

    def timeout(self, maximum: float) -> float:
        return min(maximum, self.remaining())

    def backoff(self, attempt: int) -> None:
        delay = min(0.25 * 2**attempt, self.remaining())
        time.sleep(delay)
        self.remaining()

    @contextmanager
    def enforce(self) -> Iterator[None]:
        # POSIX CLI выполняется в главном потоке. Одних httpx timeout недостаточно:
        # это лимиты сетевых фаз, которые не ограничивают медленное чтение целиком.
        if (
            not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()
        ):
            raise RuntimeError("Общий дедлайн требует POSIX и главного потока")

        def expire(signum: int, frame: object) -> None:
            raise RequestExpired

        previous_handler = signal.signal(signal.SIGALRM, expire)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, self.remaining())
        started = time.monotonic()
        try:
            yield
            self.remaining()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(0.000001, previous_timer[0] - (time.monotonic() - started)),
                    previous_timer[1],
                )
