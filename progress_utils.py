"""
Simple progress utilities for model training and prediction
"""
import time
from contextlib import contextmanager


class Timer:
    """Simple timer context manager"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        minutes = int(self.elapsed // 60)
        seconds = int(self.elapsed % 60)
        if minutes > 0:
            print(f"[TIMER] {self.name} completed in {minutes}m {seconds}s")
        else:
            print(f"[TIMER] {self.name} completed in {seconds}s")


@contextmanager
def heartbeat(message="Processing", interval=10):
    """
    Simple heartbeat context manager that prints periodic updates

    Args:
        message: Message to print
        interval: Update interval in seconds (not used in simple version)

    Usage:
        with heartbeat("Training model"):
            # long-running operation
            model.fit(X, y)
    """
    print(f"[START] {message}...")
    start_time = time.time()

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            print(f"[DONE] {message} - completed in {minutes}m {seconds}s")
        else:
            print(f"[DONE] {message} - completed in {seconds}s")
